// StochGateKernel4 -- 4-horizontally-adjacent-pixels-per-thread version of StochGateKernel.
//
// NOT a standalone header. #include it in EventGeneratorCUDA.cu, inside the anonymous
// namespace, right after the closing brace of StochGateKernel, so it can see
// stochastic_device::{Rng, init_rng, drift_diffusion}, PixelStateGPU, StochCounters
// and the PROF_* macros.
//
// Requirements (checked on the host before launching, otherwise use StochGateKernel):
//   * width  % 4 == 0
//   * stride % 4 == 0   (here stride == width)
//   * blockDim.x == 32  (one warp == 32 threads of one row; the warp-wide prefix sum
//                        and the shuffles below assume exactly that)
//   * grid.x = ceil((width / 4) / blockDim.x),  grid.y = ceil(height / blockDim.y)
//
// Differences from StochGateKernel:
//   * 1 thread handles pixels x0..x0+3 -> 4x fewer threads/blocks/warps
//   * one 8-byte image load and two 16-byte state loads/stores per thread
//   * sqrtf(dtFrameUs) once per thread instead of once per pixel
//   * one Philox init + one curand_normal4 per thread (4 normals) instead of one
//     init + one normal per pixel. Statistically equivalent, NOT bit-identical to
//     the per-pixel stream. A different seed is used so these draws are not
//     correlated with the per-pixel streams of StochEventKernel.
//   * candidate pixels: state is written back UNCHANGED (it is the same value this
//     thread loaded, and StochEventKernel runs after this kernel finishes), so the
//     vector store is safe.
//   * PROF_LAP cycle counts from this kernel are per 4-pixel thread, so the host's
//     "cycles/thread" division by numPixels under-reports by ~4x. Compare timing
//     (ms), not those cycle figures, between the two gate kernels.

__global__ void StochGateKernel4(
    PixelStateGPU *states, const uint16_t *__restrict__ image,
    int width, int height, int stride,
    float k1_over_dt_us, float k2, float k3, float k4, float k5, float k6,
    float thresholdOn, float thresholdOff, float dtFrameUs,
    float gateZ, bool gateEnabled,
    unsigned long long seed, unsigned long long frame,
    int *candList, StochCounters *ctr)
{
    using namespace stochastic_device;
    PROF_BEGIN();

    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const bool inside = (x0 < width && y < height); // NO early return: warp shuffles below
    const int idx0 = y * width + x0;                // multiple of 4 when inside
    const unsigned full = 0xffffffffu;
    const int lane = threadIdx.x & 31;

    float newVal[4] = {0.f, 0.f, 0.f, 0.f};
    float base[4] = {0.f, 0.f, 0.f, 0.f};
    float res[4] = {0.f, 0.f, 0.f, 0.f};
    float m[4] = {0.f, 0.f, 0.f, 0.f};
    float sDt[4] = {0.f, 0.f, 0.f, 0.f};
    unsigned candBits = 0; // bit j set -> pixel x0+j is a candidate

    if (inside)
    {
        // 4 x uint16 in one 8-byte load, 4 states (32 B) in two 16-byte loads
        const uint2 raw = __ldg(reinterpret_cast<const uint2 *>(
            image + static_cast<size_t>(y) * stride + x0));
        newVal[0] = static_cast<float>(raw.x & 0xffffu);
        newVal[1] = static_cast<float>(raw.x >> 16);
        newVal[2] = static_cast<float>(raw.y & 0xffffu);
        newVal[3] = static_cast<float>(raw.y >> 16);

        const float4 *sp = reinterpret_cast<const float4 *>(states + idx0);
        const float4 s01 = sp[0];
        const float4 s23 = sp[1];
        base[0] = s01.x; res[0] = s01.y;
        base[1] = s01.z; res[1] = s01.w;
        base[2] = s23.x; res[2] = s23.y;
        base[3] = s23.z; res[3] = s23.w;
        PROF_LAP(ctr, PH_G_LOAD);

        const float sqrtDt = sqrtf(dtFrameUs); // once per thread
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float c, sigma;
            drift_diffusion(newVal[j], base[j], k1_over_dt_us, k2, k3, k4, k5, k6, c, sigma);
            m[j] = c * dtFrameUs;
            sDt[j] = sigma * sqrtDt;

            const float reachUp = fmaxf(m[j], 0.f) + gateZ * sDt[j];
            const float reachDn = fmaxf(-m[j], 0.f) + gateZ * sDt[j];
            const bool cand = !gateEnabled ||
                              reachUp >= thresholdOn - res[j] ||
                              reachDn >= thresholdOff + res[j] ||
                              !isfinite(reachUp + reachDn); // NaN -> full path (it has the guard)
            if (cand)
                candBits |= (1u << j);
        }
        PROF_LAP(ctr, PH_G_DRIFT);
    }

    // ---- warp-aggregated append: each lane has 0..4 candidates ----
    const int nCand = __popc(candBits);
    int incl = nCand; // inclusive prefix sum over the 32 lanes
#pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        const int n = __shfl_up_sync(full, incl, o);
        if (lane >= o)
            incl += n;
    }
    const int warpTotal = __shfl_sync(full, incl, 31);
    int warpBase = 0;
    if (lane == 0 && warpTotal > 0)
        warpBase = static_cast<int>(atomicAdd(&ctr->numCandidates, static_cast<unsigned>(warpTotal)));
    warpBase = __shfl_sync(full, warpBase, 0);

    if (nCand)
    {
        int out = warpBase + incl - nCand;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (candBits & (1u << j))
                candList[out++] = idx0 + j;
    }
    PROF_LAP(ctr, PH_G_APPEND);

    // ---- advance the non-candidate pixels, then store all 4 states vectorized ----
    const unsigned adv = inside ? (~candBits & 0xFu) : 0u;
    if (adv)
    {
        Rng rng;
        const int threadId = y * (width >> 2) + (x0 >> 2); // unique per thread
        init_rng(&rng, seed ^ 0x9E3779B97F4A7C15ULL, threadId, frame);
        const float4 z4 = curand_normal4(&rng);
        const float z[4] = {z4.x, z4.y, z4.z, z4.w};

#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            if (adv & (1u << j))
            {
                float r = res[j] + m[j] + sDt[j] * z[j];
                r = fminf(fmaxf(r, -0.999f * thresholdOff), 0.999f * thresholdOn);
                res[j] = r;
                base[j] = newVal[j];
            }
            // candidate pixels keep their loaded (unchanged) state
        }

        float4 *sp = reinterpret_cast<float4 *>(states + idx0);
        sp[0] = make_float4(base[0], res[0], base[1], res[1]);
        sp[1] = make_float4(base[2], res[2], base[3], res[3]);
        PROF_LAP(ctr, PH_G_ADVANCE);
    }
}
