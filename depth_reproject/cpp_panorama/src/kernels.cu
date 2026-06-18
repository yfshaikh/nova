#include "kernels.cuh"

#include <cfloat>

namespace {

constexpr unsigned long long Z_EMPTY = 0xFFFFFFFFFFFFFFFFull;

inline dim3 grid2d(int w, int h, dim3 block) {
    return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
}

__global__ void kClearAccum(float4* accum, int pw, int ph) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    accum[y * pw + x] = make_float4(0.f, 0.f, 0.f, 0.f);
}

__global__ void kAccumBase(const uchar4* img, int istep, int iw, int ih,
                           const float* mapx, const float* mapy, const float* wgt,
                           const float4* cloud, int cstep, float near_drop,
                           float4* accum, int pw, int ph) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    const int pid = y * pw + x;

    const float w = wgt[pid];
    if (w <= 0.f) return;

    const float mx = mapx[pid];
    const float my = mapy[pid];
    if (mx < 0.f || my < 0.f) return;

    const int x0 = (int)floorf(mx);
    const int y0 = (int)floorf(my);
    if (x0 < 0 || y0 < 0 || x0 >= iw - 1 || y0 >= ih - 1) return;

    // Drop the near field from the base so the depth overlay isn't doubled.
    // Sample this camera's range at the (nearest) source pixel; if it's closer
    // than near_drop, skip — the overlay owns that pixel.
    if (cloud != nullptr && near_drop > 0.f) {
        const int sx = (int)(mx + 0.5f);
        const int sy = (int)(my + 0.5f);
        const float4 p = cloud[sy * cstep + sx];
        if (isfinite(p.x) && isfinite(p.y) && isfinite(p.z)) {
            const float rng = sqrtf(p.x * p.x + p.y * p.y + p.z * p.z);
            if (rng < near_drop) return;
        }
    }

    const float fx = mx - x0;
    const float fy = my - y0;
    const uchar4 c00 = img[y0 * istep + x0];
    const uchar4 c10 = img[y0 * istep + x0 + 1];
    const uchar4 c01 = img[(y0 + 1) * istep + x0];
    const uchar4 c11 = img[(y0 + 1) * istep + x0 + 1];

    const float w00 = (1.f - fx) * (1.f - fy);
    const float w10 = fx * (1.f - fy);
    const float w01 = (1.f - fx) * fy;
    const float w11 = fx * fy;

    const float b = w00 * c00.x + w10 * c10.x + w01 * c01.x + w11 * c11.x;
    const float g = w00 * c00.y + w10 * c10.y + w01 * c01.y + w11 * c11.y;
    const float r = w00 * c00.z + w10 * c10.z + w01 * c01.z + w11 * c11.z;

    float4 a = accum[pid];
    a.x += b * w;
    a.y += g * w;
    a.z += r * w;
    accum[pid] = a;
}

__global__ void kFinalizeBase(const float4* accum, uchar4* pano, int pw, int ph) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    const int pid = y * pw + x;
    const float4 a = accum[pid];
    uchar4 c;
    c.x = (unsigned char)fminf(255.f, fmaxf(0.f, a.x));
    c.y = (unsigned char)fminf(255.f, fmaxf(0.f, a.y));
    c.z = (unsigned char)fminf(255.f, fmaxf(0.f, a.z));
    c.w = 255;
    pano[pid] = c;
}

__global__ void kClearZ(unsigned long long* zbuf, int pw, int ph) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    zbuf[y * pw + x] = Z_EMPTY;
}

__global__ void kScatterOverlay(const float4* pc, int step, int iw, int ih,
                                CamExtrinsic ext, float scale, int pw, int ph,
                                float near_max, int stride, int splat,
                                unsigned long long* zbuf) {
    const int sx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    const int sy = (blockIdx.y * blockDim.y + threadIdx.y) * stride;
    if (sx >= iw || sy >= ih) return;

    const float4 p = pc[sy * step + sx];
    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) return;

    // rig = R * p_cam + t
    const float X = ext.R[0] * p.x + ext.R[1] * p.y + ext.R[2] * p.z + ext.t[0];
    const float Y = ext.R[3] * p.x + ext.R[4] * p.y + ext.R[5] * p.z + ext.t[1];
    const float Z = ext.R[6] * p.x + ext.R[7] * p.y + ext.R[8] * p.z + ext.t[2];

    const float horiz = hypotf(X, Z);
    if (horiz <= 1e-9f) return;
    const float rng = sqrtf(X * X + Y * Y + Z * Z);
    if (!isfinite(rng) || rng > near_max) return;

    const float az = atan2f(X, Z);
    const float h = Y / horiz;
    int gx = (int)lroundf(az * scale + pw * 0.5f);
    int gy = (int)lroundf(h * scale + ph * 0.5f);
    gx = ((gx % pw) + pw) % pw;
    if (gy < 0 || gy >= ph) return;

    // XYZRGBA color channel: bytes are R,G,B,A. Repack to BGRA (GL_BGRA / base order).
    const unsigned int u = __float_as_uint(p.w);
    const unsigned int R8 = u & 0xFF;
    const unsigned int G8 = (u >> 8) & 0xFF;
    const unsigned int B8 = (u >> 16) & 0xFF;
    const unsigned int bgra = B8 | (G8 << 8) | (R8 << 16) | (0xFFu << 24);

    // Monotonic depth quantization into the high 32 bits -> atomicMin = nearest.
    const float t = rng / near_max;                 // in [0,1]
    unsigned int dq = (unsigned int)(t * 4294967040.0f);
    const unsigned long long key = ((unsigned long long)dq << 32) | (unsigned long long)bgra;

    for (int dy = -splat; dy <= splat; ++dy) {
        const int yy = gy + dy;
        if (yy < 0 || yy >= ph) continue;
        for (int dx = -splat; dx <= splat; ++dx) {
            const int xx = ((gx + dx) % pw + pw) % pw;
            atomicMin(&zbuf[yy * pw + xx], key);
        }
    }
}

__global__ void kComposite(const unsigned long long* zbuf, uchar4* pano, int pw, int ph) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    const int pid = y * pw + x;
    const unsigned long long k = zbuf[pid];
    if (k == Z_EMPTY) return;
    const unsigned int bgra = (unsigned int)(k & 0xFFFFFFFFull);
    uchar4 c;
    c.x = bgra & 0xFF;
    c.y = (bgra >> 8) & 0xFF;
    c.z = (bgra >> 16) & 0xFF;
    c.w = 255;
    pano[pid] = c;
}

} // namespace

void launchClearAccum(float4* accum, int pw, int ph, cudaStream_t s) {
    const dim3 b(16, 16);
    kClearAccum<<<grid2d(pw, ph, b), b, 0, s>>>(accum, pw, ph);
}

void launchAccumBase(const uchar4* img, int istep, int iw, int ih,
                     const float* mapx, const float* mapy, const float* wgt,
                     const float4* cloud, int cstep, float near_drop,
                     float4* accum, int pw, int ph, cudaStream_t s) {
    const dim3 b(16, 16);
    kAccumBase<<<grid2d(pw, ph, b), b, 0, s>>>(img, istep, iw, ih, mapx, mapy, wgt,
                                               cloud, cstep, near_drop, accum, pw, ph);
}

void launchFinalizeBase(const float4* accum, uchar4* pano, int pw, int ph, cudaStream_t s) {
    const dim3 b(16, 16);
    kFinalizeBase<<<grid2d(pw, ph, b), b, 0, s>>>(accum, pano, pw, ph);
}

void launchClearZ(unsigned long long* zbuf, int pw, int ph, cudaStream_t s) {
    const dim3 b(16, 16);
    kClearZ<<<grid2d(pw, ph, b), b, 0, s>>>(zbuf, pw, ph);
}

void launchScatterOverlay(const float4* pc, int step, int iw, int ih,
                          CamExtrinsic ext, float scale, int pw, int ph,
                          float near_max, int stride, int splat,
                          unsigned long long* zbuf, cudaStream_t s) {
    const dim3 b(16, 16);
    const dim3 g((iw / stride + b.x - 1) / b.x, (ih / stride + b.y - 1) / b.y);
    kScatterOverlay<<<g, b, 0, s>>>(pc, step, iw, ih, ext, scale, pw, ph,
                                    near_max, stride, splat, zbuf);
}

void launchComposite(const unsigned long long* zbuf, uchar4* pano, int pw, int ph, cudaStream_t s) {
    const dim3 b(16, 16);
    kComposite<<<grid2d(pw, ph, b), b, 0, s>>>(zbuf, pano, pw, ph);
}
