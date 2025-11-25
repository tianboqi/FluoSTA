# Tianbo Qi, Sep 2025

# Edward Ntiri, 2021 Jan

# 3D structure tensor
# Code adapted from Qiyan Tian, McNab Lab, 2015 Dec

# import libraries
import argparse
import nibabel as nib
import numpy as np
import os
import sys

from scipy import signal
import scipy.ndimage
from skimage import io
# import subprocess

# from miracl.utilfn.misc import get_orient

# from dipy.tracking import utils
# from dipy.tracking.local_tracking import LocalTracking
# from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
# from dipy.direction.peaks import PeaksAndMetrics
# from dipy.direction import DeterministicMaximumDirectionGetter
# from dipy.tracking.streamline import Streamlines
# from dipy.io.streamline import save_trk
# from nibabel.affines import voxel_sizes as _vox, apply_affine
# from nibabel.streamlines import Tractogram, TrkFile, TckFile


def helpmsg():
    return '''Calculates V3 tensor from image

    Usage: python sta_flow.py -i <nifti> -g <dog sigma> -k <gaussian sigma> -b <brain mask> -s <seed> -o <output dir>

    Example: python sta_flow.py -i image.nii.gz -g 0.5 -k 2.0 -b brain_mask.nii.gz -s seed_mask.nii.gz -o g05k20

        required arguments:
            i. Input down-sampled clarity nifti (.nii/.nii.gz)
            g. Derivative of Gaussian (dog) sigma
            k. Gaussian smoothing sigma
            a. Tracking angle threshold
            b. Brain mask (.nii/.nii.gz)
            s. Seed mask (.nii/.nii.gz)

        optional arguments:
            o. Out dir

        ----------
    Main Outputs
        fiber.tck => Fiber tracts

        ----------
    Dependencies:
        - MRtrix

    -----------------------------------
    (c) Qiyuan Tian @ Stanford University, 2016
    qytian@stanford.edu
    (c) Maged Goubran @ AICONSlab, 2022
    maged.goubran@utoronto.ca
    -----------------------------------
    Modified by Tianbo @ Scripps Research, 2025
    -----------------------------------
    '''

def parsefn():
    parser = argparse.ArgumentParser(description='', usage=helpmsg())

    parser.add_argument('-i', '--input_clar', type=str,
                        help="Input down-sampled clarity nifti (.nii/.nii.gz)", required=True)
    parser.add_argument('-b', '--brainmask', type=str,
                        help="Brain mask (.nii/.nii.gz)", required=True)
    parser.add_argument('-s', '--seedmask', type=str,
                        help="Seed mask (.nii/.nii.gz)", required=True)

    # Make these numeric lists directly via argparse
    parser.add_argument('-g', '--dog', type=float, 
                        help="derivative of gaussian (dog) sigma",
                        default=[3.0, 5.0])
    parser.add_argument('-k', '--gauss', type=float, 
                        help="Gaussian smoothing sigma",
                        default=[3.0, 5.0])

    parser.add_argument('-o', '--outdir', type=str,
                        help="Output directory", default='clarity_sta')

    return parser

def parse_inputs(args):
    input_clar = args.input_clar
    brainmask = args.brainmask
    seedmask = args.seedmask
    outdir = args.outdir

    # These are already cast to the right types by argparse (lists of floats/ints)
    dog = args.dog
    gauss = args.gauss

    return input_clar, brainmask, seedmask, dog, gauss, outdir

# ---------------------------------
# functions  - main
# ---------------------------------

def doggen(sigma=None, X=None, Y=None, Z=None):
    """Function to generate derivative of Gaussian kernels, in 1D, 2D and 3D.
    
    (c) Qiyuan Tian, McNab Lab, Stanford University, September 2015

    Args:
        sigma ():
        X:
        Y:
        Z:

    Return:
        Derivative of Gaussian kernel
    """
    halfsize = np.ceil(3 * np.max(sigma))
    x = np.arange(-halfsize,halfsize+1)
    dim = len(sigma)

    if dim == 1:
        if X is None:
            X = x
        k = (-1)*X * np.exp( (-1)*np.power(X,2)/(2 * np.power(sigma[0],2)) )
    if dim == 2:
        if X is None or Y is None:
            [X,Y] =np.meshgrid(x,x)
        k = (-1)*X * np.exp( (-1)*np.power(X,2)/(2 * np.power(sigma[0],2)) ) * np.exp( (-1)*np.power(Y,2)/(2 * np.power(sigma[1],2)) )
    if dim == 3:
        if X is None or Y is None or Z is None:
            [X,Y,Z] =np.meshgrid(x,x,x)
        k = (-1)*X * np.exp( (-1)*np.power(X,2)/(2 * np.power(sigma[0],2)) ) * np.exp( (-1)*np.power(Y,2)/(2 * np.power(sigma[1],2)) ) * np.exp( (-1)*np.power(Z,2)/(2 * np.power(sigma[2],2)))
    if dim > 3:
        print ('Only support up to dimension 3')
    
    return k /np.sum(np.abs(k))

def gaussgen(sigma):
    """ Function to generate Gaussian kernels, in 1D, 2D and 3D.
    
    (c) Qiyuan Tian, McNab Lab, Stanford University, September 2015

    Args:
        sigma:

    Returns:
        Gaussian kernel
    """
    halfsize = np.ceil(3 * np.max(sigma))
    x = np.arange(-halfsize,halfsize+1)

    if len(sigma) == 1:
        k = np.exp( (-1)*np.power(X,2)/(2 * np.power(sigma[0],2)) )
    if len(sigma) == 2:
        [X,Y] =np.meshgrid(x,x)
        k = np.exp( (-1)*np.power(X,2)/(2 * np.power(sigma[0],2)) ) * np.exp( (-1)*np.power(Y,2)/(2 * np.power(sigma[1],2)) )
    if len(sigma) == 3:
        [X,Y,Z] =np.meshgrid(x,x,x)
        k = np.exp( (-1)*np.power(X,2)/(2 * np.power(sigma[0],2)) ) * np.exp( (-1)*np.power(Y,2)/(2 * np.power(sigma[1],2)) ) * np.exp( (-1)*np.power(Z,2)/(2 * np.power(sigma[2],2)))
    if len(sigma) > 3:
        print('Only support up to dimension 3')

    return k /np.sum(np.abs(k))


def gradCompute(img, dogsigma, mode='same'):
    """ Given an image as well as a dog sigma, return the gradient poduct, as well as the gradient amplitude

    """

    # derivative of gaussian kernel
    dogkercc = doggen([dogsigma, dogsigma, dogsigma])  # column
    dogkerrr = np.transpose( dogkercc, (1, 0, 2) )  # row
    dogkerzz = np.transpose( dogkercc, (0, 2, 1) )  # z-axis

    gcc = signal.convolve(img, dogkercc, mode).astype(np.float32)
    grr = signal.convolve(img, dogkerrr, mode).astype(np.float32)
    gzz = signal.convolve(img, dogkerzz, mode).astype(np.float32)

    # Gradient products
    gp = type('', (), {})()
    gp.gprrrr = grr * grr
    gp.gprrcc = grr * gcc
    gp.gprrzz = grr * gzz
    gp.gpcccc = gcc * gcc
    gp.gpcczz = gcc * gzz
    gp.gpzzzz = gzz * gzz

    # Gradient amplitude
    ga = np.sqrt(gp.gprrrr + gp.gpcccc + gp.gpzzzz).astype(np.float32)

    # Gradient vector
    gv = np.stack((grr, gcc, gzz), axis=3).astype(np.float32)
    # source for next line: https://stackoverflow.com/q/32238227
    gv = gv / np.tile(ga[..., None], [1, 1, 1, 3])  # add singleton-dim to properly tile

    return ga, gv, gp


def gradBlur(gp, gaussker, mode='same'):
    """ Blur the gradient product using the gaussian kernel
    """
    gpgauss = type('', (), {})()
    gpgauss.gprrrrgauss = signal.convolve(gp.gprrrr, gaussker, mode)
    gpgauss.gprrccgauss = signal.convolve(gp.gprrcc, gaussker, mode)
    gpgauss.gprrzzgauss = signal.convolve(gp.gprrzz, gaussker, mode)
    gpgauss.gpccccgauss = signal.convolve(gp.gpcccc, gaussker, mode)
    gpgauss.gpcczzgauss = signal.convolve(gp.gpcczz, gaussker, mode)
    gpgauss.gpzzzzgauss = signal.convolve(gp.gpzzzz, gaussker, mode)

    return gpgauss


def cropVol(volume, size):
    """ Crop the edges of the input volume by variable size, based on shape
    """
    vol = np.copy(volume)
    end = vol.shape
    dims = len(end)

    if dims == 3:
        vol[:size, :, :] = 0
        vol[end[0]-size:, :, :] = 0

        vol[:, :size, :] = 0
        vol[:, end[1]-size:, :] = 0

        vol[:, :, :size] = 0
        vol[:, :, end[2]-size:] = 0
    elif dims == 4:
        vol[:size, :, :, :] = 0
        vol[end[0]-size:, :, :, :] = 0

        vol[:, :size, :, :] = 0
        vol[:, end[1]-size:, :, :] = 0

        vol[:, :, :size, :] = 0
        vol[:, :, end[2]-size:, :] = 0
    else:
        raise ValueError('Variable had {} dimensions, expected either 3 or 4.')

    return vol


def generateAffine(vox_size=[1,1,1]):
    ''' Generate affine for NIFTI image
    '''
    affine = np.eye(4)
    affine[0,0] = vox_size[0]
    affine[1,1] = vox_size[1]
    affine[2,2] = vox_size[2]

    return affine


# --- helper to write clean NIfTI ---
def save_float32(path, arr, hdr, affine):
    hdr['scl_slope'] = 1
    hdr['scl_inter'] = 0
    ni = nib.Nifti1Image(arr.astype(np.float32, copy=False), affine, hdr)
    nib.save(ni, path)

def save_uint8_mask(path, arr_bool, hdr, affine):
    hdr['scl_slope'] = 1
    hdr['scl_inter'] = 0
    ni = nib.Nifti1Image(arr_bool.astype(np.uint8, copy=False), affine, hdr)
    nib.save(ni, path)

# ---------------------------------
# read and compute structure tensor
# ---------------------------------
def sta_track(stack, dog_sigma, gauss_sigma, fpBmask, fp_smask, dpResult):
    """
    Main function to compute structure tensor and save v3 and coherence
    Tianbo Qi, 2025
    """

    out_dir = os.path.join(dpResult, f'dog{dog_sigma}gau{gauss_sigma}')
    if not os.path.exists(out_dir):
        print('Creating directory:', out_dir)
        os.makedirs(out_dir)
    
    fsl_path = os.path.join(out_dir, 'fsl_tensor.nii.gz')
    fp_bmask_crop = os.path.join(out_dir, 'bmask.nii.gz')
    fp_smask_crop = os.path.join(out_dir, 'smask.nii.gz')
    out_v3 = os.path.join(out_dir, 'v3.nii.gz')
    out_coh = os.path.join(out_dir, 'coherence.nii.gz')

    # gradient kernel
    dogkercc = doggen([dog_sigma, dog_sigma, dog_sigma])
    # smoothing kernel
    gaussker = gaussgen([gauss_sigma, gauss_sigma, gauss_sigma])
    # half kernel size used by your cropVol
    halfsize = int((max(gaussker.shape[0], dogkercc.shape[0]) + 1) / 2)

    if not os.path.exists(fsl_path):

        # --- read image (float32; keeps memory sane) ---
        print("Reading image...")
        img = nib.load(stack)
        img_data = img.get_fdata(dtype=np.float32)
        img_affine = img.affine
        img_hdr = img.header.copy()
        X, Y, Z = img_data.shape

        # (Optional) normalize for numerical stability
        # img_data = (img_data - img_data.min()) / (img_data.ptp() + 1e-8)

        # vox_order = get_orient(stack)  # not used

        # compute gradient amplitude, vector, product
        print("Computing gradients...")
        ga, gv, gp = gradCompute(img_data, dog_sigma)

        grad_path = os.path.join(out_dir, 'grad_unsmoothed.nii.gz')
        grad_tensor = np.stack([
            gp.gprrrr,
            gp.gprrcc,
            gp.gpcccc,
            gp.gprrzz,
            gp.gpcczz,
            gp.gpzzzz
        ], axis=3)
        save_float32(grad_path, grad_tensor, img_hdr.copy(), img_affine)
        print("original gradient saved.")

        # --- save GA / GV if you want them on disk (unchanged logic) ---
        # ga_path = os.path.join(out_dir, 'ga.nii')
        # if not os.path.exists(ga_path):
        #     save_float32(ga_path, cropVol(ga, halfsize), img_hdr.copy(), img_affine)
        # else:
        #     print("ga.nii.gz already exists. Skipping...")

        # gv_path = os.path.join(out_dir, 'gv.nii')
        # if not os.path.exists(gv_path):
        #     save_float32(gv_path, cropVol(gv, halfsize), img_hdr.copy(), img_affine)
        # else:
        #     print("gv.nii.gz already exists. Skipping...")

        # --- blur gradient product ---
        print("Smoothing gradients...")
        gp_gauss = gradBlur(gp, gaussker)
            
        # crop & save fsl tensor (float32, no scaling)

        fsl_tensor = np.stack([
            gp_gauss.gprrrrgauss,  # Dxx
            gp_gauss.gprrccgauss,  # Dxy
            gp_gauss.gprrzzgauss,  # Dxz
            gp_gauss.gpccccgauss,  # Dyy
            gp_gauss.gpcczzgauss,  # Dyz
            gp_gauss.gpzzzzgauss   # Dzz
        ], axis=3)
        save_float32(fsl_path, cropVol(fsl_tensor, halfsize), img_hdr.copy(), img_affine)
        print("fsl_tensor.nii.gz saved.")

        Dxx = gp_gauss.gprrrrgauss  # Dxx
        Dxy = gp_gauss.gprrccgauss  # Dxy
        Dxz = gp_gauss.gprrzzgauss  # Dxz
        Dyy = gp_gauss.gpccccgauss  # Dyy
        Dyz = gp_gauss.gpcczzgauss  # Dyz
        Dzz = gp_gauss.gpzzzzgauss  # Dzz

        # # crop & save dtk tensor (float32, no scaling)
        # dtk_path = os.path.join(out_dir, 'dtk_tensor.nii.gz')
        # if not os.path.exists(dtk_path):
        #     dtk_tensor = np.stack([
        #         gp_gauss.gprrrrgauss,
        #         gp_gauss.gprrccgauss,
        #         gp_gauss.gpccccgauss,
        #         gp_gauss.gprrzzgauss,
        #         gp_gauss.gpcczzgauss,
        #         gp_gauss.gpzzzzgauss
        #     ], axis=3)
        #     save_float32(dtk_path, cropVol(dtk_tensor, halfsize), img_hdr.copy(), img_affine)
        # else:
        #     print("dtk_tensor.nii already exists. Skipping...")

        # --- crop & save masks on same grid as cropped image ---
    
    else:
        print("fsl tensor already exists. Reading...")
        # --- read image (float32; keeps memory sane) ---
        fsl_img = nib.load(fsl_path)
        fsl_ten = fsl_img.get_fdata(dtype=np.float32)
        img = nib.load(stack)
        img_data = img.get_fdata(dtype=np.float32)
        img_affine = img.affine
        img_hdr = img.header.copy()
        X, Y, Z = img_data.shape

        Dxx, Dxy, Dxz, Dyy, Dyz, Dzz = [fsl_ten[...,i] for i in range(6)]

    # --- build full (X,Y,Z,3,3) symmetric tensor and eigendecompose in one go ---
    D = np.empty((X, Y, Z, 3, 3), dtype=np.float32)
    D[..., 0, 0] = Dxx
    D[..., 1, 1] = Dyy
    D[..., 2, 2] = Dzz
    D[..., 0, 1] = D[..., 1, 0] = Dxy
    D[..., 0, 2] = D[..., 2, 0] = Dxz
    D[..., 1, 2] = D[..., 2, 1] = Dyz

    # np.linalg.eigh returns eigenvalues ascending and eigenvectors as columns
    # shapes: l -> (X,Y,Z,3), V -> (X,Y,Z,3,3)
    print("Calculating eigenvectors...")
    l, V = np.linalg.eigh(D)

    # v3 = eigenvector of the smallest eigenvalue = column 0
    v3 = V[..., :, 0]                                   # (X,Y,Z,3)
    v3 /= (np.linalg.norm(v3, axis=-1, keepdims=True) + 1e-8)
    v3 = v3.astype(np.float32, copy=False)
    
    save_float32(out_v3, v3, img_hdr.copy(), img_affine)
    print("v3 tensor saved!")

    coherence = (l[:,:,:,2] - l[:,:,:,1]) / (l.sum(axis=-1) + 1e-8)
    save_float32(out_coh, coherence, img_hdr.copy(), img_affine)
    print("Coherence tensor saved!")
        
            
    if not os.path.exists(fp_bmask_crop):
        bmask = nib.load(fpBmask).get_fdata(dtype=np.float32)
        bmask = cropVol(bmask, halfsize) > 0.5
        save_uint8_mask(fp_bmask_crop, bmask, img_hdr.copy(), img_affine)
        print('bmask.nii.gz saved.')
    else:
        print("bmask.nii.gz already exists. Skipping...")

    
    if not os.path.exists(fp_smask_crop):
        smask = nib.load(fp_smask).get_fdata(dtype=np.float32)
        smask = cropVol(smask, halfsize) > 0.5
        save_uint8_mask(fp_smask_crop, smask, img_hdr.copy(), img_affine)
        print('smask.nii.gz saved.')
    else:
        print("smask.nii.gz already exists. Skipping...")


# ---------------------------------
# set parameters and filenames
# ---------------------------------

def main(argv=None):
    parser = parsefn()
    args = parser.parse_args(argv)
    (filename, bmask, smask, dog_sigma, gauss_sigma, output_dir) = parse_inputs(args)

    # run sta tract generation
    print('Running Structure Tensor Analysis\n')
    sta_track(filename, dog_sigma, gauss_sigma, bmask, smask, output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
