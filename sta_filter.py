#!/usr/bin/env python3
import argparse
import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine

def helpmsg():
    return """
        Prune streamlines based on suffix-mean intensity / coherence values and optional angle trimming.
        All pruning operates on a per-streamline basis. Output is written as a new .tck file.

        USAGE:
        python sta_filter.py --in_tck IN.tck --img IMG.nii.gz --out_tck OUT.tck
                            [--threshold THR] [--coh COH.nii.gz --coh_threshold THR]
                            [--angle_max DEG]
                            [--interp tri|nn]
                            [--nan_policy ignore|zero|fail]
                            [--coh_nan_policy ignore|zero|fail]
                            [--img_min_tail_valid N]
                            [--coh_min_tail_valid N]

        REQUIRED:
        --in_tck PATH          Input tractogram (.tck)
        --img PATH             Image used for IMAGE pruning (RASMM space)
        --out_tck PATH         Output tractogram (.tck)

        IMAGE PRUNING (suffix mean over sampled image values):
        --threshold FLOAT      Prune at earliest index where suffix-mean of IMAGE values
                                falls below this threshold. NaNs handled per --nan_policy.
        --img_min_tail_valid N Minimum number of valid samples in tail (default 1).
        --nan_policy MODE      Handling of NaNs for IMAGE pruning:
                                ignore : exclude NaNs from mean
                                zero   : treat NaNs as zero & count them
                                fail   : any NaN causes streamline to be dropped

        COHERENCE PRUNING (suffix mean):
        --coh PATH             Optional coherence image (.nii/.nii.gz). If omitted,
                                reuse --img for coherence pruning.
        --coh_threshold FLOAT  As above, but for COHERENCE image.
        --coh_min_tail_valid N Minimum valid tail samples for coherence (default 1).
        --coh_nan_policy MODE  NaN policy for coherence pruning (ignore/zero/fail).

        ANGLE TRIMMING:
        --angle_max DEG        Trim streamline at the first turn exceeding this angle
                                (in degrees) in world (RASMM) space.
                                Angles computed from consecutive segments.

        INTERPOLATION:
        --interp tri|nn        tri = trilinear interpolation (default)
                                nn  = nearest-neighbor sampling

        NOTES:
        * All streamline coordinates are interpreted in RASMM world space.
        * Sampling coordinates are mapped into voxel space using each image's affine.
        * Suffix-mean pruning returns earliest point i where:
                mean(vals[i:]) < threshold
            (subject to NaN policy and tail valid-count).
        * If multiple pruning modes are active, order is:
                1) coherence → 2) image → 3) angle trim
        * A streamline is kept only if ≥ 2 points remain after all pruning.

        EXAMPLES:
        Basic image pruning:
            python prune_tck.py --in_tck fibers.tck --img FA.nii.gz \
                --threshold 0.2 --out_tck pruned.tck

        Coherence pruning + angle trimming:
            python prune_tck.py --in_tck fibers.tck --img FA.nii.gz \
                --coh cohmap.nii.gz --coh_threshold 0.1 --angle_max 45 \
                --out_tck pruned.tck

"""


# ---------- sampling helpers ----------
def sample_trilinear(vol, pts_vox):
    X, Y, Z = vol.shape
    x, y, z = pts_vox[:, 0], pts_vox[:, 1], pts_vox[:, 2]
    valid = (x >= 0) & (x < X - 1) & (y >= 0) & (y < Y - 1) & (z >= 0) & (z < Z - 1)
    out = np.full(pts_vox.shape[0], np.nan, dtype=np.float32)
    if not np.any(valid):
        return out
    xv = x[valid]; yv = y[valid]; zv = z[valid]
    x0 = np.floor(xv).astype(np.int64)
    y0 = np.floor(yv).astype(np.int64)
    z0 = np.floor(zv).astype(np.int64)
    dx = xv - x0; dy = yv - y0; dz = zv - z0
    x1 = x0 + 1; y1 = y0 + 1; z1 = z0 + 1
    c000 = vol[x0, y0, z0]
    c100 = vol[x1, y0, z0]
    c010 = vol[x0, y1, z0]
    c110 = vol[x1, y1, z0]
    c001 = vol[x0, y0, z1]
    c101 = vol[x1, y0, z1]
    c011 = vol[x0, y1, z1]
    c111 = vol[x1, y1, z1]
    c00 = c000 * (1 - dx) + c100 * dx
    c01 = c001 * (1 - dx) + c101 * dx
    c10 = c010 * (1 - dx) + c110 * dx
    c11 = c011 * (1 - dx) + c111 * dx
    c0  = c00 * (1 - dy) + c10 * dy
    c1  = c01 * (1 - dy) + c11 * dy
    vals = c0 * (1 - dz) + c1 * dz
    out[valid] = vals.astype(np.float32, copy=False)
    return out

def sample_nearest(vol, pts_vox):
    X, Y, Z = vol.shape
    x = np.rint(pts_vox[:, 0]).astype(np.int64)
    y = np.rint(pts_vox[:, 1]).astype(np.int64)
    z = np.rint(pts_vox[:, 2]).astype(np.int64)
    valid = (x >= 0) & (x < X) & (y >= 0) & (y < Y) & (z >= 0) & (z < Z)
    out = np.full(pts_vox.shape[0], np.nan, dtype=np.float32)
    out[valid] = vol[x[valid], y[valid], z[valid]].astype(np.float32, copy=False)
    return out

# ---------- suffix-mean pruning ----------
def first_suffix_mean_below(vals, thresh, min_tail_valid=1, mode="ignore"):
    """
    Find the smallest index i such that the mean of the tail vals[i:]
    is < thresh. Tail mean is computed per 'mode':
      - "ignore": mean over finite values only (NaNs excluded from count).
      - "zero":   NaNs treated as 0 and COUNTED in the mean.
      - "fail":   return -2 to signal 'invalid' if any NaN is present.
    Also require at least 'min_tail_valid' valid samples in the tail
    (for "zero", all samples are counted as valid).
    Returns:
      - i >= 0 : prune at i (keep pts[:i])
      - None   : no pruning
      - -2     : mode='fail' and found NaN -> caller should drop streamline
    """
    vals = np.asarray(vals, dtype=np.float64)
    n = vals.size
    if n == 0:
        return None

    if mode == "fail":
        if np.isnan(vals).any():
            return -2  # signal drop
        # fall-through to "ignore" semantics since no NaNs
        mode = "ignore"

    if mode == "zero":
        valid_cnt = np.ones(n, dtype=np.int32)
        vals_filled = np.where(np.isfinite(vals), vals, 0.0)
    else:  # "ignore"
        finite = np.isfinite(vals)
        valid_cnt = finite.astype(np.int32)
        vals_filled = np.where(finite, vals, 0.0)

    suffix_sum = np.zeros(n, dtype=np.float64)
    suffix_cnt = np.zeros(n, dtype=np.int32)

    run_sum = 0.0
    run_cnt = 0
    for i in range(n - 1, -1, -1):
        run_sum += vals_filled[i]
        run_cnt += valid_cnt[i]
        suffix_sum[i] = run_sum
        suffix_cnt[i] = run_cnt

    # need sufficient tail count AND mean < thresh  => sum < thresh * cnt
    cond = (suffix_cnt >= int(min_tail_valid)) & (suffix_sum < float(thresh) * suffix_cnt)
    if not np.any(cond):
        return None
    return int(np.argmax(cond))  # earliest i where condition holds

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description='', usage=helpmsg())
    ap.add_argument("--img", required=True, help="Image (.nii/.nii.gz) used for IMAGE pruning (RASMM space)")
    ap.add_argument("--out_tck", required=True, help="Output tractogram (.tck)")

    # image pruning (suffix mean)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Prune at earliest point where the suffix-mean of IMAGE values falls below this threshold.")
    ap.add_argument("--img_min_tail_valid", type=int, default=1,
                    help="Minimum valid samples in the tail for image pruning (default 1).")

    # angle trimming
    ap.add_argument("--angle_max", type=float, default=None,
                    help="Trim at first turn whose angle (deg) exceeds this value (in world space).")

    # coherence pruning (suffix mean)
    ap.add_argument("--coh", default=None,
                    help="Coherence image (.nii/.nii.gz). If omitted, reuse --img.")
    ap.add_argument("--coh_threshold", type=float, default=None,
                    help="Prune at earliest point where the suffix-mean of COHERENCE falls below this value.")
    ap.add_argument("--coh_min_tail_valid", type=int, default=1,
                    help="Minimum valid samples in the tail for coherence pruning (default 1).")

    # interpolation & NaN policies
    ap.add_argument("--interp", choices=["tri", "nn"], default="tri", help="Interpolation for sampling.")
    ap.add_argument("--nan_policy", choices=["ignore", "fail", "zero"], default="ignore",
                    help="How to treat NaNs for IMAGE pruning: ignore/zero/fail.")
    ap.add_argument("--coh_nan_policy", choices=["ignore", "fail", "zero"], default="ignore",
                    help="How to treat NaNs for COHERENCE pruning.")

    args = ap.parse_args()

    # Load main image (IMAGE)
    img = nib.load(args.img)
    vol = np.asanyarray(img.dataobj).astype(np.float32, copy=False)
    Ainv = np.linalg.inv(img.affine)

    # Load coherence image (or reuse IMAGE)
    if args.coh is not None:
        coh_img = nib.load(args.coh)
        coh_vol = np.asanyarray(coh_img.dataobj).astype(np.float32, copy=False)
        coh_Ainv = np.linalg.inv(coh_img.affine)
    else:
        coh_img = img
        coh_vol = vol
        coh_Ainv = Ainv

    # Load TCK
    tck_obj = nib.streamlines.load(args.in_tck)
    tract = tck_obj.tractogram
    streamlines = list(tract.streamlines)

    if args.threshold is None and args.angle_max is None and args.coh_threshold is None:
        raise SystemExit("Nothing to do: set at least one of --threshold, --coh_threshold, --angle_max.")

    sample_fn = sample_trilinear if args.interp == "tri" else sample_nearest

    kept = []
    n_img_prune = n_coh_prune = n_ang_trim = 0

    for sl in streamlines:
        if sl.shape[0] < 2:
            continue

        # ----- COHERENCE pruning (suffix mean) -----
        if args.coh_threshold is not None:
            pts_vox_coh = apply_affine(coh_Ainv, sl.astype(np.float64, copy=False))
            coh_vals = sample_fn(coh_vol, pts_vox_coh)
            cut_i = first_suffix_mean_below(
                coh_vals, args.coh_threshold, args.coh_min_tail_valid, mode=args.coh_nan_policy
            )
            if cut_i == -2:   # fail mode and NaN present
                continue
            if cut_i is not None:
                if cut_i <= 1:
                    continue
                sl = sl[:cut_i]
                n_coh_prune += 1

        # ----- IMAGE pruning (suffix mean) -----
        if args.threshold is not None and sl.shape[0] >= 2:
            pts_vox_img = apply_affine(Ainv, sl.astype(np.float64, copy=False))
            img_vals = sample_fn(vol, pts_vox_img)
            cut_i = first_suffix_mean_below(
                img_vals, args.threshold, args.img_min_tail_valid, mode=args.nan_policy
            )
            if cut_i == -2:   # fail mode and NaN present
                continue
            if cut_i is not None:
                if cut_i <= 1:
                    continue
                sl = sl[:cut_i]
                n_img_prune += 1

        # ----- Angle trimming (world/RASMM) -----
        if args.angle_max is not None and sl.shape[0] >= 3:
            pts = sl.astype(np.float64, copy=False)
            V = np.diff(pts, axis=0)
            n = np.linalg.norm(V, axis=1)
            valid = n > 1e-12
            idx_valid = np.nonzero(valid)[0]
            if idx_valid.size >= 2:
                U = V[valid] / n[valid][:, None]
                dots = np.sum(U[:-1] * U[1:], axis=1)
                np.clip(dots, -1.0, 1.0, out=dots)
                angs = np.degrees(np.arccos(dots))
                bad = np.nonzero(angs > float(args.angle_max))[0]
                if bad.size > 0:
                    cut_idx = idx_valid[bad[0] + 1] + 1  # segment->point index
                    sl = pts[:cut_idx]
                    n_ang_trim += 1
            if sl.shape[0] < 2:
                continue

        kept.append(sl)

    print(f"Loaded {len(streamlines)} streamlines.")
    if args.coh_threshold is not None:
        print(f"  Coherence prune (suffix-mean): pruned {n_coh_prune} "
              f"(threshold={args.coh_threshold}, min_tail_valid={args.coh_min_tail_valid}, policy={args.coh_nan_policy})")
    if args.threshold is not None:
        print(f"  Image prune (suffix-mean): pruned {n_img_prune} "
              f"(threshold={args.threshold}, min_tail_valid={args.img_min_tail_valid}, policy={args.nan_policy})")
    if args.angle_max is not None:
        print(f"  Angle trim: trimmed {n_ang_trim} streamlines at turns > {args.angle_max}°")
    print(f"Final kept: {len(kept)}")

    out_tg = nib.streamlines.Tractogram(kept, affine_to_rasmm=tract.affine_to_rasmm)
    nib.streamlines.save(out_tg, args.out_tck)
    print(f"Saved -> {args.out_tck}")

if __name__ == "__main__":
    main()
