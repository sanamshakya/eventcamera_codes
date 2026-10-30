#pragma once
// Diagnostic "floor" kernels: isolate how much of the gate kernel's time is
// memory traffic vs launch/scheduling, by moving a controlled number of bytes
// per pixel and doing no other work.
//
// TIMING ONLY. These kernels overwrite d_pixelStates_ with junk and emit no
// events. Include this header from exactly one .cu file.
//
// Modes (bytes moved per pixel):
//   0 : 2 B read, 8 B write  (same traffic as the existing gateZ<0 floor)
//   1 : 2 B read, 2 B write
//   2 : 1 B read, 1 B write
//   3 : 1 B read, no write   (read cost alone)
//   4 : no read, 1 B write   (write cost alone)
//   5 : empty kernel         (launch + block scheduling only)
//   6 : 1 B read, 1 B write, 4 pixels per thread via 32-bit load/store
//       (needs width % 4 == 0)
//
// Modes 2, 3, 6 read the first width*height bytes of the uint16 image buffer
// and modes 2, 4, 6 write the first width*height bytes of the state buffer;
// both are large enough, the values are meaningless, only the traffic matters.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace evsim
{

__global__ void FloorTestKernel(int mode, const uint16_t *__restrict__ image,
                                void *states, int width, int height, int stride)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    const int idx = y * width + x;

    switch (mode)
    {
    case 0:
    {
        const float v = static_cast<float>(image[static_cast<size_t>(y) * stride + x]);
        reinterpret_cast<float2 *>(states)[idx] = make_float2(v, 0.f);
        break;
    }
    case 1:
        reinterpret_cast<uint16_t *>(states)[idx] = image[static_cast<size_t>(y) * stride + x];
        break;
    case 2:
        reinterpret_cast<uint8_t *>(states)[idx] =
            reinterpret_cast<const uint8_t *>(image)[idx];
        break;
    case 3:
    {
        const uint8_t v = reinterpret_cast<const uint8_t *>(image)[idx];
        // Never true at runtime (x < width), but the compiler cannot know that,
        // so the load cannot be optimized away.
        if (v == 0xA5 && x == 0x7fffffff)
            reinterpret_cast<uint8_t *>(states)[0] = v;
        break;
    }
    case 4:
        reinterpret_cast<uint8_t *>(states)[idx] = static_cast<uint8_t>(x ^ y);
        break;
    default: // 5: nothing
        break;
    }
}

// 4 pixels per thread: one 32-bit load and one 32-bit store per thread.
__global__ void FloorTestVec4Kernel(const uint16_t *__restrict__ image, void *states,
                                    int width, int height)
{
    const int x4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x4 >= width || y >= height)
        return;
    const int idx = y * width + x4; // multiple of 4 when width % 4 == 0
    const uint32_t v = reinterpret_cast<const uint32_t *>(
        reinterpret_cast<const uint8_t *>(image) + idx)[0];
    reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(states) + idx)[0] = v;
}

// Grid is computed from the ACTUAL block shape, so it covers exactly the image.
inline void launchFloorTest(int mode, const uint16_t *image, void *states,
                            int width, int height, int stride, cudaStream_t stream)
{
    const dim3 block(32, 16);
    if (mode == 6)
    {
        const dim3 grid((width / 4 + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        FloorTestVec4Kernel<<<grid, block, 0, stream>>>(image, states, width, height);
    }
    else
    {
        const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        FloorTestKernel<<<grid, block, 0, stream>>>(mode, image, states, width, height, stride);
    }
}

} // namespace evsim
