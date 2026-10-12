#pragma once
//
// ClaheCuda - CUDA implementation of CLAHE (contrast limited adaptive
// histogram equalization) for single-channel images whose values live in
// the range 0 .. (2^bits - 1).

//
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace evsim
{

class ClaheCuda
{
public:
    ClaheCuda(int width, int height,
              float clipLimit = 2.0f,
              int tilesX = 8, int tilesY = 8,
              int bits = 8, int inputShift = 0);
    ~ClaheCuda();

    ClaheCuda(const ClaheCuda&)            = delete;
    ClaheCuda& operator=(const ClaheCuda&) = delete;

    // Cheap: only reallocates the LUT buffer if the tile grid changes.
    void setParams(float clipLimit, int tilesX, int tilesY);

    // bits must be 8 or 12 (anything else is clamped to the nearest).
    // Reallocates the LUT only if the bit depth changes.
    void setFormat(int bits, int inputShift);

    // Device pointers; pitches in BYTES (pass width * sizeof(pixel) when
    // tightly packed). src and dst must be different buffers.
    //
    // apply()    : uint8_t pixels. Only meaningful with bits == 8.
    // applyU16() : uint16_t pixels, any configured bit depth.
    void apply(const uint8_t* d_src, size_t srcPitch,
               uint8_t*       d_dst, size_t dstPitch,
               cudaStream_t stream = 0);

    void applyU16(const uint16_t* d_src, size_t srcPitch,
                  uint16_t*       d_dst, size_t dstPitch,
                  cudaStream_t stream = 0);

    int   width()      const { return w_; }
    int   height()     const { return h_; }
    float clipLimit()  const { return clip_; }
    int   tilesX()     const { return tx_; }
    int   tilesY()     const { return ty_; }
    int   bits()       const { return bits_; }
    int   inputShift() const { return shift_; }

private:
    template <typename T>
    void run(const T* d_src, size_t srcPitch, T* d_dst, size_t dstPitch, cudaStream_t stream);

    void allocLut();
    void freeLut();

    int   w_, h_;
    int   tx_, ty_;
    float clip_;
    int   bits_;
    int   shift_;
    uint16_t* d_lut_ = nullptr;   // tx_ * ty_ * (1 << bits_) entries
};

} // namespace evsim
