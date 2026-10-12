#include "ClaheCuda.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace evsim
{

#define CLAHE_CUDA_CHECK(call)                                                  \
    do {                                                                        \
        cudaError_t e_ = (call);                                                \
        if (e_ != cudaSuccess) {                                                \
            fprintf(stderr, "ClaheCuda CUDA error %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(e_));                \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

namespace
{

//----------------------------------------------------------------------
// Kernel 1: one block (THREADS threads) per tile.
// Histogram -> clip -> redistribute -> CDF -> LUT, all in shared memory.
// Each thread owns PER = BINS/THREADS contiguous histogram bins.
//   8-bit : BINS=256,  THREADS=256 (PER=1)
//   12-bit: BINS=4096, THREADS=512 (PER=8)  -> 16 KB hist + 2 KB sums
//----------------------------------------------------------------------
template <typename T, int BINS, int THREADS>
__global__ void claheLutKernel(const T* __restrict__ src,
                               size_t srcPitch,
                               int w, int h,
                               int tileW, int tileH,
                               float clipLimit,
                               int inShift,
                               uint16_t* __restrict__ lut)
{
    constexpr int PER = BINS / THREADS;
    __shared__ int hist[BINS];
    __shared__ int sums[THREADS];
    __shared__ int excess;

    const int tid = threadIdx.x;
    const int x0  = blockIdx.x * tileW;
    const int y0  = blockIdx.y * tileH;
    const int tw  = min(tileW, w - x0);   // last tile may be smaller
    const int th  = min(tileH, h - y0);
    const int tileArea = tw * th;

    for (int b = tid; b < BINS; b += THREADS) hist[b] = 0;
    if (tid == 0) excess = 0;
    __syncthreads();

    // 1) histogram
    const uint8_t* base = reinterpret_cast<const uint8_t*>(src);
    for (int i = tid; i < tileArea; i += THREADS)
    {
        const int yy = i / tw;
        const int xx = i - yy * tw;
        const T* row = reinterpret_cast<const T*>(base + (size_t)(y0 + yy) * srcPitch);
        const int v = min((int)(row[x0 + xx] >> inShift), BINS - 1);
        atomicAdd(&hist[v], 1);
    }
    __syncthreads();

    // 2) clip + redistribute (same integer math as OpenCV)
    const bool doClip = clipLimit > 0.0f;
    if (doClip)
    {
        const int clip = max((int)(clipLimit * (float)tileArea / (float)BINS), 1);
        int mine = 0;
        #pragma unroll
        for (int j = 0; j < PER; ++j)
        {
            int &c = hist[tid * PER + j];
            if (c > clip) { mine += c - clip; c = clip; }
        }
        if (mine) atomicAdd(&excess, mine);
    }
    __syncthreads();

    if (doClip)
    {
        const int ex       = excess;
        const int batch    = ex / BINS;
        const int residual = ex - batch * BINS;
        const int step     = residual > 0 ? max(BINS / residual, 1) : 1;
        #pragma unroll
        for (int j = 0; j < PER; ++j)
        {
            const int b = tid * PER + j;
            int add = batch;
            if (residual > 0 && (b % step) == 0 && (b / step) < residual)
                add += 1;
            hist[b] += add;
        }
    }
    __syncthreads();

    // 3) CDF: local inclusive prefix over own bins, then Hillis-Steele scan
    //    over the per-thread totals.
    int loc[PER];
    int run = 0;
    #pragma unroll
    for (int j = 0; j < PER; ++j) { run += hist[tid * PER + j]; loc[j] = run; }
    sums[tid] = run;
    __syncthreads();
    for (int off = 1; off < THREADS; off <<= 1)
    {
        const int v = (tid >= off) ? sums[tid - off] : 0;
        __syncthreads();
        sums[tid] += v;
        __syncthreads();
    }
    const int offset = sums[tid] - run;   // exclusive prefix of earlier threads

    // 4) LUT
    const float lutScale = (float)(BINS - 1) / (float)tileArea;
    uint16_t* out = lut + (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * BINS;
    #pragma unroll
    for (int j = 0; j < PER; ++j)
    {
        int o = __float2int_rn((float)(offset + loc[j]) * lutScale);
        out[tid * PER + j] = (uint16_t)min(max(o, 0), BINS - 1);
    }
}

//----------------------------------------------------------------------
// Kernel 2: per-pixel bilinear interpolation between 4 neighbouring LUTs.
//----------------------------------------------------------------------
template <typename T, int BINS>
__global__ void claheApplyKernel(const T* __restrict__ src,
                                 size_t srcPitch,
                                 T* __restrict__ dst,
                                 size_t dstPitch,
                                 int w, int h,
                                 int tileW, int tileH,
                                 int tilesX, int tilesY,
                                 int inShift,
                                 const uint16_t* __restrict__ lut)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const float txf = (float)x / (float)tileW - 0.5f;
    int   tx1 = (int)floorf(txf);
    int   tx2 = tx1 + 1;
    const float xa  = txf - (float)tx1;
    const float xa1 = 1.0f - xa;
    tx1 = max(tx1, 0);
    tx2 = min(tx2, tilesX - 1);

    const float tyf = (float)y / (float)tileH - 0.5f;
    int   ty1 = (int)floorf(tyf);
    int   ty2 = ty1 + 1;
    const float ya  = tyf - (float)ty1;
    const float ya1 = 1.0f - ya;
    ty1 = max(ty1, 0);
    ty2 = min(ty2, tilesY - 1);

    const T* srow = reinterpret_cast<const T*>(
        reinterpret_cast<const uint8_t*>(src) + (size_t)y * srcPitch);
    const int v = min((int)(srow[x] >> inShift), BINS - 1);

    const float l11 = lut[(size_t)(ty1 * tilesX + tx1) * BINS + v];
    const float l12 = lut[(size_t)(ty1 * tilesX + tx2) * BINS + v];
    const float l21 = lut[(size_t)(ty2 * tilesX + tx1) * BINS + v];
    const float l22 = lut[(size_t)(ty2 * tilesX + tx2) * BINS + v];

    const float res = (l11 * xa1 + l12 * xa) * ya1 +
                      (l21 * xa1 + l22 * xa) * ya;

    int o = __float2int_rn(res);
    o = min(max(o, 0), BINS - 1);

    T* drow = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(dst) + (size_t)y * dstPitch);
    drow[x] = static_cast<T>(o);
}

} // anonymous namespace

//----------------------------------------------------------------------
// Host side
//----------------------------------------------------------------------
static int normalizeBits(int bits) { return bits <= 8 ? 8 : 12; }

ClaheCuda::ClaheCuda(int width, int height, float clipLimit, int tilesX, int tilesY,
                     int bits, int inputShift)
    : w_(width), h_(height), tx_(tilesX), ty_(tilesY), clip_(clipLimit),
      bits_(normalizeBits(bits)), shift_(inputShift < 0 ? 0 : inputShift)
{
    allocLut();
}

ClaheCuda::~ClaheCuda()
{
    freeLut();
}

void ClaheCuda::allocLut()
{
    CLAHE_CUDA_CHECK(cudaMalloc(&d_lut_,
        (size_t)tx_ * ty_ * ((size_t)1 << bits_) * sizeof(uint16_t)));
}

void ClaheCuda::freeLut()
{
    if (d_lut_) { cudaFree(d_lut_); d_lut_ = nullptr; }
}

void ClaheCuda::setParams(float clipLimit, int tilesX, int tilesY)
{
    clip_ = clipLimit;
    if (tilesX != tx_ || tilesY != ty_)
    {
        // Make sure no in-flight kernel is still using the old LUT.
        CLAHE_CUDA_CHECK(cudaDeviceSynchronize());
        freeLut();
        tx_ = tilesX;
        ty_ = tilesY;
        allocLut();
    }
}

void ClaheCuda::setFormat(int bits, int inputShift)
{
    shift_ = inputShift < 0 ? 0 : inputShift;
    const int nb = normalizeBits(bits);
    if (nb != bits_)
    {
        CLAHE_CUDA_CHECK(cudaDeviceSynchronize());
        freeLut();
        bits_ = nb;
        allocLut();
    }
}

template <typename T>
void ClaheCuda::run(const T* d_src, size_t srcPitch, T* d_dst, size_t dstPitch,
                    cudaStream_t stream)
{
    const int tileW = (w_ + tx_ - 1) / tx_;
    const int tileH = (h_ + ty_ - 1) / ty_;
    const dim3 lutGrid(tx_, ty_);

    if (bits_ == 8)
    {
        claheLutKernel<T, 256, 256><<<lutGrid, 256, 0, stream>>>(
            d_src, srcPitch, w_, h_, tileW, tileH, clip_, shift_, d_lut_);
    }
    else
    {
        claheLutKernel<T, 4096, 512><<<lutGrid, 512, 0, stream>>>(
            d_src, srcPitch, w_, h_, tileW, tileH, clip_, shift_, d_lut_);
    }

    const dim3 block(32, 8);
    const dim3 grid((w_ + block.x - 1) / block.x,
                    (h_ + block.y - 1) / block.y);

    if (bits_ == 8)
    {
        claheApplyKernel<T, 256><<<grid, block, 0, stream>>>(
            d_src, srcPitch, d_dst, dstPitch,
            w_, h_, tileW, tileH, tx_, ty_, shift_, d_lut_);
    }
    else
    {
        claheApplyKernel<T, 4096><<<grid, block, 0, stream>>>(
            d_src, srcPitch, d_dst, dstPitch,
            w_, h_, tileW, tileH, tx_, ty_, shift_, d_lut_);
    }

    CLAHE_CUDA_CHECK(cudaGetLastError());
}

void ClaheCuda::apply(const uint8_t* d_src, size_t srcPitch,
                      uint8_t* d_dst, size_t dstPitch, cudaStream_t stream)
{
    run<uint8_t>(d_src, srcPitch, d_dst, dstPitch, stream);
}

void ClaheCuda::applyU16(const uint16_t* d_src, size_t srcPitch,
                         uint16_t* d_dst, size_t dstPitch, cudaStream_t stream)
{
    run<uint16_t>(d_src, srcPitch, d_dst, dstPitch, stream);
}

} // namespace evsim
