#include "ResizeRaw.cuh"

__device__ __forceinline__ float boxL(const uint16_t *src, size_t pitchElems,
                                      int w, int h, int x, int y)
{
    const int x1 = min(x + 1, w - 1);
    const int y1 = min(y + 1, h - 1);
    const uint16_t *r0 = src + size_t(y) * pitchElems;
    const uint16_t *r1 = src + size_t(y1) * pitchElems;
    return 0.25f * (float(r0[x]) + float(r0[x1]) + float(r1[x]) + float(r1[x1]));
}

__global__ void resizeRawU16Kernel(const uint16_t *src, size_t pitchElems, int sw, int sh,
                                   uint16_t *dst, int dw, int dh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh)
        return;

    const float sx = fminf(fmaxf((x + 0.5f) * sw / dw - 0.5f, 0.f), sw - 1.f);
    const float sy = fminf(fmaxf((y + 0.5f) * sh / dh - 0.5f, 0.f), sh - 1.f);
    const int x0 = int(sx), y0 = int(sy);
    const int x1 = min(x0 + 1, sw - 1), y1 = min(y0 + 1, sh - 1);
    const float fx = sx - x0, fy = sy - y0;

    const float v =
        (1.f - fy) * ((1.f - fx) * boxL(src, pitchElems, sw, sh, x0, y0) +
                      fx         * boxL(src, pitchElems, sw, sh, x1, y0)) +
        fy         * ((1.f - fx) * boxL(src, pitchElems, sw, sh, x0, y1) +
                      fx         * boxL(src, pitchElems, sw, sh, x1, y1));

    dst[size_t(y) * dw + x] = static_cast<uint16_t>(v + 0.5f);
}

cudaError_t launchResizeRawU16(const uint16_t *src, size_t srcPitchBytes, int sw, int sh,
                               uint16_t *dst, int dw, int dh)
{
    const dim3 block(32, 8);
    const dim3 grid((dw + block.x - 1) / block.x, (dh + block.y - 1) / block.y);
    resizeRawU16Kernel<<<grid, block>>>(src, srcPitchBytes / sizeof(uint16_t), sw, sh, dst, dw, dh);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        return e;
    // The generator runs on its own stream, so synchronise explicitly.
    return cudaDeviceSynchronize();
}