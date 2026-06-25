#pragma once
//
// CUDA kernels for the GPU-end-to-end 360 panorama.
//   Base layer : per-camera inverse-warp (cylinder remap) + feather blend.
//   Overlay    : forward-scatter depth reprojection with a packed uint64
//                z-buffer (depth<<32 | BGRA) resolved by a single atomicMin.
//   Composite  : overlay over base, straight into the CUDA-GL PBO (uchar4 BGRA).
//
// All buffers live on the GPU for the whole frame; nothing touches the CPU.
//
#include <cuda_runtime.h>

// Row-major cam->rig rotation (R) + translation (t), in meters.
// rig = R * p_cam + t   (matches Python cam_points_to_rig: P_cam @ R.T + t)
struct CamExtrinsic {
    float R[9];
    float t[3];
};

// accum is a float4 scratch buffer (xyz = weighted BGR sum, w unused).
void launchClearAccum(float4* accum, int pano_w, int pano_h, cudaStream_t s);

// Bilinearly sample one camera's BGRA image through its precomputed cylinder
// map (mapx/mapy, pano-sized) and add color * weight into accum.
//
// If cloud != nullptr and near_drop > 0, base pixels whose source depth is
// closer than near_drop (meters) are SKIPPED. The depth overlay redraws that
// near field depth-correct from the rig origin, so leaving it in the rotation-
// only base would double it (the smeared parallax copy under the overlay).
void launchAccumBase(const uchar4* img, int img_step, int img_w, int img_h,
                     const float* mapx, const float* mapy, const float* wgt,
                     const float4* cloud, int cloud_step, float near_drop,
                     float4* accum, int pano_w, int pano_h, cudaStream_t s);

// accum -> pano (uchar4 BGRA, alpha 255).
void launchFinalizeBase(const float4* accum, uchar4* pano,
                        int pano_w, int pano_h, cudaStream_t s);

// Reset the packed z-buffer to "empty" (all bits set = +inf depth).
void launchClearZ(unsigned long long* zbuf, int pano_w, int pano_h, cudaStream_t s);

// Forward-scatter one camera's XYZRGBA cloud into the shared z-buffer.
// stride subsamples the source; splat writes an (2r+1)^2 neighborhood to seal
// pinholes (replaces the CPU morphology in the Python version).
void launchScatterOverlay(const float4* pc, int pc_step, int img_w, int img_h,
                          CamExtrinsic ext, float scale, int pano_w, int pano_h,
                          float near_max, int stride, int splat,
                          unsigned long long* zbuf, cudaStream_t s);

// Write the nearest overlay color over the base where the z-buffer is populated.
void launchComposite(const unsigned long long* zbuf, uchar4* pano,
                     int pano_w, int pano_h, cudaStream_t s);
