#!/usr/bin/env python3

# Tianbo Qi, Sep 2025
# STA tracking using v3 direction field

# import libraries

import argparse
import numpy as np
import nibabel as nib

try:
    from numba import njit, prange
except Exception:
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap

# Only used for seeding; pure Python, no Cython core
from dipy.tracking import utils as dutils
from nibabel.streamlines import Tractogram, save as save_tractogram_nb


def helpmsg():
    return '''Fast STA tractography along a single direction field (v3) from sta_flow.py

    Usage: python sta_track.py --v3 <v3image> --seeds <smask from sta_flow> --mask <bmask from sta_flow> --out <output tck file> --step <step size mm> --angle <angle threshold> 
                              [--coherence <coherence image> --cmin <coherence threshold>] [--density <seed density> --max_steps <max track steps> --min_steps <min track steps>]

    Example: python sta_track.py --v3 v3.nii.gz --seeds smask.nii.gz --mask bmask.nii.gz --out streamlines.tck --step 1e-3 --angle 35 --density 0.1

    - Follows voxelwise unit vector field (v3) with nearest-neighbor sampling.
    - Enforces max turning angle (deg) online.
    - Stops at brain mask and optional coherence threshold.
    - Seeds from a seed mask (DIPY's seeds_from_mask).
    - Saves streamlines to .tck (RASMM) via NiBabel.

    -----------------------------------
    Tianbo @ Scripps Research, 2025
    -----------------------------------
    '''



# -------------------- I/O --------------------
def load_nii(path):
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    return data, img.affine, img


# -------------------- Math helpers (Numba) --------------------
@njit(cache=True, fastmath=True)
def _dot3(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@njit(cache=True, fastmath=True)
def _norm3(a):
    return np.sqrt(_dot3(a, a))

@njit(cache=True, fastmath=True)
def _normalize3(a):
    n = _norm3(a)
    if n <= 0.0 or not np.isfinite(n):
        return 0, np.zeros(3, dtype=np.float64)
    return 1, a / n

@njit(cache=True, fastmath=True)
def _matvec33(A33, v):
    return np.array([A33[0,0]*v[0] + A33[0,1]*v[1] + A33[0,2]*v[2],
                     A33[1,0]*v[0] + A33[1,1]*v[1] + A33[1,2]*v[2],
                     A33[2,0]*v[0] + A33[2,1]*v[1] + A33[2,2]*v[2]], dtype=np.float64)

@njit(cache=True, fastmath=True)
def _apply_affine_4x4(A, p):  # world <- voxel or voxel <- world (depending on A)
    x = A[0,0]*p[0] + A[0,1]*p[1] + A[0,2]*p[2] + A[0,3]
    y = A[1,0]*p[0] + A[1,1]*p[1] + A[1,2]*p[2] + A[1,3]
    z = A[2,0]*p[0] + A[2,1]*p[1] + A[2,2]*p[2] + A[2,3]
    return np.array([x, y, z], dtype=np.float64)

@njit(cache=True, fastmath=True)
def _clamp_round(a, lo, hi):
    # nearest-neighbor: round to nearest integer index and clamp to [lo, hi]
    i = int(np.round(a))
    if i < lo: i = lo
    if i > hi: i = hi
    return i

@njit(cache=True, fastmath=True)
def _sample_vec_nn(vfield, X, Y, Z, xv, yv, zv):
    # vfield: (X,Y,Z,3) float32
    if xv < 0 or yv < 0 or zv < 0 or xv > X-1 or yv > Y-1 or zv > Z-1:
        return 0, np.zeros(3, dtype=np.float64)
    xi = _clamp_round(xv, 0, X-1)
    yi = _clamp_round(yv, 0, Y-1)
    zi = _clamp_round(zv, 0, Z-1)
    vv = np.array([vfield[xi, yi, zi, 0],
                   vfield[xi, yi, zi, 1],
                   vfield[xi, yi, zi, 2]], dtype=np.float64)
    if vv[0] == 0.0 and vv[1] == 0.0 and vv[2] == 0.0:
        return 0, vv
    return 1, vv

@njit(cache=True, fastmath=True)
def _sample_vec_tri(vfield, X, Y, Z, xv, yv, zv):
    """
    Trilinear interpolation of a 3D vector field.
    vfield: (X, Y, Z, 3) float32
    (xv, yv, zv): voxel-space floating coords
    Returns: (ok:int, vec:float64[3])
      ok = 1 if in-bounds and interpolated vector not all-zero, else 0
    """
    # Need one neighbor in +x/+y/+z for trilinear
    if xv < 0.0 or yv < 0.0 or zv < 0.0 or xv > X - 2 or yv > Y - 2 or zv > Z - 2:
        return 0, np.zeros(3, dtype=np.float64)

    x0 = int(np.floor(xv))
    y0 = int(np.floor(yv))
    z0 = int(np.floor(zv))
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    dx = xv - x0
    dy = yv - y0
    dz = zv - z0
    omdx = 1.0 - dx
    omdy = 1.0 - dy
    omdz = 1.0 - dz

    # Interpolate each component independently (Numba-friendly, no tiny arrays)
    # ---- x component ----
    c000 = vfield[x0, y0, z0, 0]; c100 = vfield[x1, y0, z0, 0]
    c010 = vfield[x0, y1, z0, 0]; c110 = vfield[x1, y1, z0, 0]
    c001 = vfield[x0, y0, z1, 0]; c101 = vfield[x1, y0, z1, 0]
    c011 = vfield[x0, y1, z1, 0]; c111 = vfield[x1, y1, z1, 0]
    c00 = c000 * omdx + c100 * dx
    c10 = c010 * omdx + c110 * dx
    c01 = c001 * omdx + c101 * dx
    c11 = c011 * omdx + c111 * dx
    cx  = (c00 * omdy + c10 * dy) * omdz + (c01 * omdy + c11 * dy) * dz

    # ---- y component ----
    c000 = vfield[x0, y0, z0, 1]; c100 = vfield[x1, y0, z0, 1]
    c010 = vfield[x0, y1, z0, 1]; c110 = vfield[x1, y1, z0, 1]
    c001 = vfield[x0, y0, z1, 1]; c101 = vfield[x1, y0, z1, 1]
    c011 = vfield[x0, y1, z1, 1]; c111 = vfield[x1, y1, z1, 1]
    c00 = c000 * omdx + c100 * dx
    c10 = c010 * omdx + c110 * dx
    c01 = c001 * omdx + c101 * dx
    c11 = c011 * omdx + c111 * dx
    cy  = (c00 * omdy + c10 * dy) * omdz + (c01 * omdy + c11 * dy) * dz

    # ---- z component ----
    c000 = vfield[x0, y0, z0, 2]; c100 = vfield[x1, y0, z0, 2]
    c010 = vfield[x0, y1, z0, 2]; c110 = vfield[x1, y1, z0, 2]
    c001 = vfield[x0, y0, z1, 2]; c101 = vfield[x1, y0, z1, 2]
    c011 = vfield[x0, y1, z1, 2]; c111 = vfield[x1, y1, z1, 2]
    c00 = c000 * omdx + c100 * dx
    c10 = c010 * omdx + c110 * dx
    c01 = c001 * omdx + c101 * dx
    c11 = c011 * omdx + c111 * dx
    cz  = (c00 * omdy + c10 * dy) * omdz + (c01 * omdy + c11 * dy) * dz

    vv = np.array([cx, cy, cz], dtype=np.float64)

    norm = np.sqrt(vv[0]*vv[0] + vv[1]*vv[1] + vv[2]*vv[2])
    vv = vv/norm

    if vv[0] == 0.0 and vv[1] == 0.0 and vv[2] == 0.0:
        return 0, vv
    return 1, vv


@njit(cache=True, fastmath=True)
def _sample_mask_nn(mask, X, Y, Z, xv, yv, zv):
    if xv < 0 or yv < 0 or zv < 0 or xv > X-1 or yv > Y-1 or zv > Z-1:
        return 0
    xi = _clamp_round(xv, 0, X-1)
    yi = _clamp_round(yv, 0, Y-1)
    zi = _clamp_round(zv, 0, Z-1)
    return 1 if mask[xi, yi, zi] != 0 else 0

@njit(cache=True, fastmath=True)
def _sample_scalar_nn(vol, X, Y, Z, xv, yv, zv):
    if xv < 0 or yv < 0 or zv < 0 or xv > X-1 or yv > Y-1 or zv > Z-1:
        return 0, 0.0
    xi = _clamp_round(xv, 0, X-1)
    yi = _clamp_round(yv, 0, Y-1)
    zi = _clamp_round(zv, 0, Z-1)
    return 1, float(vol[xi, yi, zi])


# -------------------- Tracking core (Numba) --------------------
@njit(cache=True, fastmath=True)
def _trace_one_side(seed_w, sign, vfield, mask, coherence, have_coh, cmin,
                    Ainv, A33, step_mm, max_angle_rad, max_steps):
    """
    Follow vfield from a seed, single direction (sign = +1 or -1).
    Returns filled length and array (max_steps+1, 3) float32 in world coords.
    """
    X, Y, Z, _ = vfield.shape
    max_steps = int(max_steps)
    pts = np.zeros((max_steps + 1, 3), dtype=np.float32)
    pts[0, 0] = seed_w[0]; pts[0, 1] = seed_w[1]; pts[0, 2] = seed_w[2]
    npts = 1

    prev_dir_w = np.zeros(3, dtype=np.float64)
    have_prev = False

    for _ in range(max_steps):
        p_w = np.array([pts[npts - 1, 0], pts[npts - 1, 1], pts[npts - 1, 2]], dtype=np.float64)
        p_v = _apply_affine_4x4(Ainv, p_w)

        ok, vv = _sample_vec_nn(vfield, X, Y, Z, p_v[0], p_v[1], p_v[2])
        # ok, vv = _sample_vec_tri(vfield, X, Y, Z, p_v[0], p_v[1], p_v[2])
        if ok == 0:
            print("Ended with invalid vector")
            break

        # direction in world coordinates
        v_signed = np.array([sign * vv[0], sign * vv[1], sign * vv[2]], dtype=np.float64)
        d_w = _matvec33(A33, v_signed)
        ok2, d_w = _normalize3(d_w)
        if ok2 == 0:
            print("Ended with invalid vector")
            break

        if have_prev:
            # Antipodal alignment: make the new dir face roughly the same way as the previous
            if _dot3(prev_dir_w, d_w) < 0.0:
                d_w = -d_w
            cosang = _dot3(prev_dir_w, d_w)
            # both unit
            if cosang < -1.0: cosang = -1.0
            if cosang >  1.0: cosang =  1.0
            ang = np.arccos(cosang)
            if ang > max_angle_rad:
                print("Ended with large turning angle")
                break

        next_w = np.array([p_w[0] + step_mm * d_w[0],
                           p_w[1] + step_mm * d_w[1],
                           p_w[2] + step_mm * d_w[2]], dtype=np.float64)

        # stop if outside mask or below coherence threshold
        next_v = _apply_affine_4x4(Ainv, next_w)
        if _sample_mask_nn(mask, X, Y, Z, next_v[0], next_v[1], next_v[2]) == 0:
            print("Ended outside the mask")
            break
        if have_coh:
            ok3, val = _sample_scalar_nn(coherence, X, Y, Z, next_v[0], next_v[1], next_v[2])
            if ok3 == 0 or val < cmin:
                print("Ended with low coherence")
                break

        pts[npts, 0] = np.float32(next_w[0])
        pts[npts, 1] = np.float32(next_w[1])
        pts[npts, 2] = np.float32(next_w[2])
        npts += 1
        prev_dir_w = d_w
        have_prev = True

        if _ == max_steps - 1:
            print("Ended with max steps")

    return npts, pts


@njit(cache=True, fastmath=True)
def _track_seed(seed_w, vfield, mask, coherence, have_coh, cmin,
                Ainv, A33, step_mm, max_angle_rad, max_steps):
    # backward
    nB, B = _trace_one_side(seed_w, -1, vfield, mask, coherence, have_coh, cmin,
                            Ainv, A33, step_mm, max_angle_rad, max_steps)
    # forward
    nF, F = _trace_one_side(seed_w, +1, vfield, mask, coherence, have_coh, cmin,
                            Ainv, A33, step_mm, max_angle_rad, max_steps)

    # # merge (drop duplicate seed)
    # total = nB + nF - 1
    # out = np.zeros((total, 3), dtype=np.float32)
    # # reverse B
    # for i in range(nB - 1, -1, -1):
    #     out[nB - 1 - i, 0] = B[i, 0]
    #     out[nB - 1 - i, 1] = B[i, 1]
    #     out[nB - 1 - i, 2] = B[i, 2]
    # # append F (skip first)
    # idx = nB
    # for i in range(1, nF):
    #     out[idx, 0] = F[i, 0]
    #     out[idx, 1] = F[i, 1]
    #     out[idx, 2] = F[i, 2]
    #     idx += 1
    # return total, out
    return nB, B, nF, F

# @njit(parallel=True, cache=True, fastmath=True)
# def track_all_seeds_parallel(seeds_w, vfield, mask, coherence, have_coh, cmin,
#                              Ainv, A33, step_mm, max_angle_rad, max_steps):
#     n = seeds_w.shape[0]
#     # Preallocate [n, max_steps*2+1, 3] (back+fwd merged worst-case)
#     max_steps = int(max_steps)
#     max_pts = max_steps * 2 + 1
#     out_xyz = np.zeros((n, max_pts, 3), dtype=np.float32)
#     out_len = np.zeros(n, dtype=np.int32)

#     for i in prange(n):
#         n_i, sl_i = _track_seed(
#             seed_w=seeds_w[i].astype(np.float64),
#             vfield=vfield, mask=mask, coherence=coherence, have_coh=have_coh, cmin=cmin,
#             Ainv=Ainv, A33=A33, step_mm=step_mm,
#             max_angle_rad=max_angle_rad, max_steps=max_steps,
#             # include any extra guards you added:
#             # e.g., max_length_mm, stuck_voxel_limit
#         )
#         # Clip to buffer
#         # if n_i > max_pts:
#         #     n_i = max_pts
#         out_xyz[i, :n_i, :] = sl_i[:n_i, :]
#         out_len[i] = n_i

#     return out_xyz, out_len

@njit(parallel=True, cache=True, fastmath=True)
def track_all_seeds_parallel(seeds_w, vfield, mask, coherence, have_coh, cmin,
                                      Ainv, A33, step_mm, max_angle_rad, max_steps):
    """
    Returns a single set of arrays representing a ragged list of streamlines:
      out_xyz_all: (n*2, half_max_pts, 3)
      out_len_all: (n*2,)
    Entry 2*i   is the backward half-tract for seed i
    Entry 2*i+1 is the forward  half-tract for seed i
    """
    n  = seeds_w.shape[0]
    ms = int(max_steps)
    half_max_pts = ms + 1

    out_xyz_all = np.zeros((n * 2, half_max_pts, 3), dtype=np.float32)
    out_len_all = np.zeros(n * 2, dtype=np.int32)

    for i in prange(n):
        nB, B, nF, F = _track_seed(
            seed_w=seeds_w[i].astype(np.float64),
            vfield=vfield, mask=mask, coherence=coherence, have_coh=have_coh, cmin=cmin,
            Ainv=Ainv, A33=A33, step_mm=step_mm,
            max_angle_rad=max_angle_rad, max_steps=ms
        )

        # clip defensively
        if nB > half_max_pts: nB = half_max_pts
        if nF > half_max_pts: nF = half_max_pts

        bi = 2 * i
        fi = bi + 1

        out_xyz_all[bi, :nB, :] = B[:nB, :]
        out_len_all[bi] = nB

        out_xyz_all[fi, :nF, :] = F[:nF, :]
        out_len_all[fi] = nF

    return out_xyz_all, out_len_all

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description='', usage=helpmsg())
    ap.add_argument("--v3", required=True, help="v3 direction field (X,Y,Z,3) NIfTI; unit vectors")
    ap.add_argument("--seeds", required=True, help="Seed mask NIfTI (binary)")
    ap.add_argument("--mask", required=True, help="Brain mask NIfTI (binary)")
    ap.add_argument("--out", required=True, help="Output .tck")

    ap.add_argument("--coherence", default=None, help="Optional coherence NIfTI")
    ap.add_argument("--cmin", type=float, default=None, help="Minimum coherence to continue (stop if below)")

    ap.add_argument("--step", type=float, default=0.005, help="Step size in mm")
    ap.add_argument("--angle", type=float, default=90.0, help="Max turning angle (deg)")
    ap.add_argument("--density", type=float, default=1, help="Seeds per voxel edge (1 => ~1 seed/voxel)")
    ap.add_argument("--max_steps", type=int, default=5e3, help="Max steps per half-tract")
    ap.add_argument("--min_steps", type=int, default=0, help="Discard streamlines with fewer steps")
    args = ap.parse_args()

    # Load volumes
    v3, affine, ref_img = load_nii(args.v3)          # (X,Y,Z,3)
    seeds_mask, _, _ = load_nii(args.seeds)          # (X,Y,Z)
    brain_mask, _, _ = load_nii(args.mask)           # (X,Y,Z)
    print("Images loaded")

    if v3.ndim != 4 or v3.shape[-1] != 3:
        raise ValueError(f"v3 must be 4D with last dim=3; got {v3.shape}")
    if v3.shape[:3] != seeds_mask.shape[:3] or v3.shape[:3] != brain_mask.shape[:3]:
        raise ValueError("Spatial shapes of v3, seed mask, and brain mask must match")

    # Clean and gate
    # v3 = np.nan_to_num(v3.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    # vnorm = np.linalg.norm(v3, axis=-1)
    # v3[vnorm == 0] = 0.0
    brain_mask = (brain_mask > 0).astype(np.uint8)
    v3[~(brain_mask.astype(bool))] = 0.0
    seeds_mask = (seeds_mask > 0).astype(np.uint8)

    if args.coherence is not None:
        coherence, _, _ = load_nii(args.coherence)
        if coherence.shape[:3] != v3.shape[:3]:
            raise ValueError("coherence volume shape must match v3 spatial shape")
        coherence = np.nan_to_num(coherence.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        if args.cmin is None:
            raise ValueError("If --coherence is provided, please set --cmin as the stopping threshold")
        have_coh = True
        cmin = float(args.cmin)
    else:
        have_coh = False
        coherence = np.zeros(v3.shape[:3], dtype=np.float32)
        cmin = 0.0

    # Seeds (RASMM/world); force array (float64, C-contig, writable)
    if args.density >= 1:
        seeds = dutils.seeds_from_mask(seeds_mask, affine, density=density)
    else:
        N = int(args.density * seeds_mask.sum())
        where = np.array(np.where(seeds_mask)).T
        sel = np.random.choice(where.shape[0], size=N, replace=False)
        voxels = where[sel]
        seeds_vox = voxels.astype(float) + 0.5
        seeds = nib.affines.apply_affine(affine, seeds_vox)
    seeds = np.array(seeds, dtype=np.float64, order="C", copy=True)

    # Affines for numba
    Ainv = np.linalg.inv(affine).astype(np.float64)
    A33 = affine[:3, :3].astype(np.float64)
    print("Starting tracking...")

    # Track each seed
    # max_angle_rad = np.deg2rad(float(args.angle))
    # streamlines = []
    # for i, s in enumerate(seeds):
    #     n, sl = _track_seed(
    #         seed_w=s.astype(np.float64),
    #         vfield=v3,
    #         mask=brain_mask,
    #         coherence=coherence,
    #         have_coh=have_coh,
    #         cmin=cmin,
    #         Ainv=Ainv,
    #         A33=A33,
    #         step_mm=float(args.step),
    #         max_angle_rad=max_angle_rad,
    #         max_steps=int(args.max_steps),
    #     )
    #     if n >= args.min_steps:
    #         streamlines.append(sl[:n].copy())
        # print(f"Tracked {i} streamlines")
    
    # # -------- Track in parallel START --------
    out_xyz, out_len = track_all_seeds_parallel(
        seeds, v3, brain_mask, coherence, have_coh, cmin,
        Ainv, A33, step_mm=args.step,
        max_angle_rad=np.deg2rad(args.angle),
        max_steps=args.max_steps
    )
    # Build ragged list
    streamlines = [out_xyz[i, :out_len[i]].copy() for i in range(out_len.shape[0]) if out_len[i] >= args.min_steps]
    # -------- Track in parallel END --------

    # Save to .tck (RASMM)
    tg = Tractogram(streamlines, affine_to_rasmm=np.eye(4, dtype=np.float64))
    save_tractogram_nb(tg, args.out)
    print(f"Saved {len(streamlines)} streamlines -> {args.out}")


if __name__ == "__main__":
    main()
