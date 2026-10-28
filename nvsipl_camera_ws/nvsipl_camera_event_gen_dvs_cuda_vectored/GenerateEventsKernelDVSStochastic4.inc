// GenerateEventsKernelDVSStochastic4 -- 4-horizontally-adjacent-pixels-per-thread version of
// GenerateEventsKernelDVSStochastic (same signature, same per-pixel math and RNG streams).
//
// NOT a standalone header. #include it in EventGeneratorCUDA.cu inside the anonymous
// namespace, right after GenerateEventsKernelDVSStochastic (it needs stochastic_device::*,
// kMaxCrossingsPerPixel, PixelStateGPU, Event, kHeaderBytes).
//
// Requirements (checked on the host before launching, otherwise use the scalar kernel):
//   * width % 4 == 0   (stride == width here, so image rows are 8-byte aligned)
//   * grid.x = ceil((width / 4) / blockDim.x),  grid.y = ceil(height / blockDim.y)
//   * blockDim.x == 32 is NOT required (no warp shuffles in this kernel)
//
// Design notes:
//   * The per-pixel curandState array is kept as-is (one XORWOW state per pixel, 48 B each),
//     so every pixel keeps exactly the same random stream as in the scalar kernel and results
//     are statistically/bit-wise consistent with it. Only ONE rng is live at a time (pixels are
//     processed sequentially inside the thread), so register pressure stays close to the
//     scalar kernel's.
//   * Vectorized: one 8-byte image load, two 16-byte state loads, two 16-byte state stores.
//   * Event slots are still claimed with one atomicAdd per emitted event (0-3 events/pixel is
//     typical, so contention is small; a warp-aggregated version is not worth it here).

namespace stochastic_device
{

// One pixel's full stochastic update. Identical to the body of the scalar kernel, factored
// out so the 4-pixel kernel can call it in a loop. Returns the new residual voltage.
__device__ __forceinline__ double simulate_pixel(
    double newVal, double baseVal, double deltaVdLegacy,
    float k1, float k2, float k3, float k4, float k5, float k6,
    float thresholdOn, float thresholdOff,
    float dtFrameUs, double dtFrameSeconds, double frameStartTime,
    int x, int y, int maxEvents, int *eventCount, Event *events,
    curandState *rng)
{
    const double deltaLight = newVal - baseVal;
    const double avgLight = (newVal + baseVal) * 0.5;
    const double denom = 1.0 / (avgLight + static_cast<double>(k2));

    const double muClean = (static_cast<double>(k1) / static_cast<double>(dtFrameUs)) * deltaLight * denom;
    const double c = muClean + static_cast<double>(k4) + static_cast<double>(k5) * avgLight; // drift
    const double varClean = (static_cast<double>(k3) * sqrt(fmax(avgLight, 0.0))) * denom;
    const double sigma = varClean + static_cast<double>(k6); // diffusion

    const double epOn = thresholdOn;
    const double epOff = thresholdOff;

    double startT = 0.0;
    const double endT = static_cast<double>(dtFrameUs);

    for (int iter = 0; iter < kMaxCrossingsPerPixel; ++iter)
    {
        const double epOnReal = epOn - deltaVdLegacy;
        const double epOffReal = epOff + deltaVdLegacy;

        const double pOn = first_on_probability(epOnReal, epOffReal, c, sigma);
        const bool on = curand_uniform_double(rng) <= pOn; // Bernoulli trial

        const double epInput = on ? epOnReal : epOffReal;
        const double cInput = on ? c : -c;
        const double deltaT = sample_IG(epInput, cInput, sigma, rng); // IG or Levy

        // NaN / non-positive guard (see scalar kernel for the rationale)
        if (!isfinite(deltaT) || deltaT <= 0.0)
            break;

        const double tEndIdeal = deltaT + startT;

        if (tEndIdeal >= endT)
        {
            const double sign = on ? 1.0 : -1.0;
            deltaVdLegacy += sign * epInput * (endT - startT) / deltaT;
            break;
        }

        const int slot = atomicAdd(eventCount, 1);
        if (slot < maxEvents)
        {
            const double frac = tEndIdeal / endT;
            const double eventTime = frameStartTime + frac * dtFrameSeconds;
            events[slot] = Event{static_cast<uint16_t>(x), static_cast<uint16_t>(y),
                                 eventTime, static_cast<int8_t>(on ? 1 : -1)};
        }

        startT = tEndIdeal;
        deltaVdLegacy = 0.0;
    }
    return deltaVdLegacy;
}

} // namespace stochastic_device

__global__ void GenerateEventsKernelDVSStochastic4(
    PixelStateGPU *states, curandState *rngStates,
    const uint16_t *__restrict__ image, int width, int height, int stride,
    float k1, float k2, float k3, float k4, float k5, float k6,
    float thresholdOn, float thresholdOff,
    float dtFrameUs, double dtFrameSeconds, double frameStartTime,
    int maxEvents, uint8_t *outputBuffer)
{
    using namespace stochastic_device;

    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x0 >= width || y >= height) // safe: no warp-wide primitives below
        return;
    const int idx0 = y * width + x0; // multiple of 4

    // 4 x uint16 in one 8-byte load, 4 states (32 B) in two 16-byte loads
    const uint2 raw = __ldg(reinterpret_cast<const uint2 *>(
        image + static_cast<size_t>(y) * stride + x0));
    const float newVal[4] = {static_cast<float>(raw.x & 0xffffu), static_cast<float>(raw.x >> 16),
                             static_cast<float>(raw.y & 0xffffu), static_cast<float>(raw.y >> 16)};

    const float4 *sp = reinterpret_cast<const float4 *>(states + idx0);
    const float4 s01 = sp[0];
    const float4 s23 = sp[1];
    const float base[4] = {s01.x, s01.z, s23.x, s23.z};
    float res[4] = {s01.y, s01.w, s23.y, s23.w};

    int *eventCount = reinterpret_cast<int *>(outputBuffer);
    Event *events = reinterpret_cast<Event *>(outputBuffer + kHeaderBytes);

    // Pixels are processed one after another so only one 48-byte curandState is live at a time.
#pragma unroll 1
    for (int j = 0; j < 4; ++j)
    {
        curandState rng = rngStates[idx0 + j]; // load once, work on local copy, store back
        res[j] = static_cast<float>(simulate_pixel(
            static_cast<double>(newVal[j]), static_cast<double>(base[j]),
            static_cast<double>(res[j]),
            k1, k2, k3, k4, k5, k6, thresholdOn, thresholdOff,
            dtFrameUs, dtFrameSeconds, frameStartTime,
            x0 + j, y, maxEvents, eventCount, events, &rng));
        rngStates[idx0 + j] = rng;
    }

    // vectorized state store: baseIntensity = newVal, deltaVdRes = updated residual
    float4 *spw = reinterpret_cast<float4 *>(states + idx0);
    spw[0] = make_float4(newVal[0], res[0], newVal[1], res[1]);
    spw[1] = make_float4(newVal[2], res[2], newVal[3], res[3]);
}

/* ---------------------------------------------------------------------------
   Host side, in EventGeneratorCUDA::generate(), replacing the stochastic launch:

        const bool vec4 = (width_ % 4 == 0);
        const dim3 grid4(((width_ / 4) + block.x - 1) / block.x,
                         (height_ + block.y - 1) / block.y);

        if (vec4)
            GenerateEventsKernelDVSStochastic4<<<grid4, block, 0, stream_>>>(...same args...);
        else
            GenerateEventsKernelDVSStochastic<<<grid, block, 0, stream_>>>(...same args...);

   Notes:
   * RNGInitKernel is unchanged (still one curandState per pixel).
   * Because this kernel is dominated by double-precision math (exp/erf/erfinv/sqrt) and
     divergent per-pixel loops, expect a smaller gain than for the deterministic kernel.
     The 4 pixels of a thread run back-to-back, so a warp's time per slot j is set by its
     slowest lane for that pixel.
   * If you want a bigger reduction (and don't need the per-pixel streams to match), use ONE
     Philox/XORWOW state per thread (like StochGateKernel4 does with curand_normal4/init_rng):
     the rngStates buffer shrinks 4x and the 4 pixels share one stream. Say so and I'll do it.
--------------------------------------------------------------------------- */
