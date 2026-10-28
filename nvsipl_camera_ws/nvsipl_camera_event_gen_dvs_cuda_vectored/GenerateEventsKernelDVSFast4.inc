// GenerateEventsKernelDVSFast4 -- 4-horizontally-adjacent-pixels-per-thread version of
// GenerateEventsKernelDVSFast (same signature, same math, same event semantics).
//
// NOT a standalone header. #include it in EventGeneratorCUDA.cu inside the anonymous
// namespace, right after GenerateEventsKernelDVSFast (it needs PixelStateGPU, Event,
// kHeaderBytes).
//
// Requirements (checked on the host before launching, otherwise use the scalar kernel):
//   * width % 4 == 0   (stride == width here, so image rows are 8-byte aligned)
//   * blockDim.x == 32 (one warp == 32 threads of one row; prefix sum assumes it)
//   * grid.x = ceil((width / 4) / blockDim.x),  grid.y = ceil(height / blockDim.y)
//
// Differences from the scalar kernel:
//   * 1 thread handles pixels x0..x0+3 -> 4x fewer threads/blocks/warps
//   * one 8-byte image load, two 16-byte state loads and two 16-byte state stores
//   * the warp-aggregated atomicAdd now sums crossings over 4 pixels per lane
//     (128 pixels per warp) -> 4x fewer atomics on the event counter
//   * slot assignment: per-thread exclusive offset from the warp prefix sum, then
//     pixels j=0..3 claim consecutive slots, so events stay ordered by x within a
//     thread (same relative order as the scalar kernel within a warp segment).

__global__ void GenerateEventsKernelDVSFast4(
    PixelStateGPU *states, const uint16_t *__restrict__ image,
    int width, int height, int stride,
    float k1_over_dt_us, float k2, float k4, float k5,
    float thresholdOn, float thresholdOff,
    float dtFrameUs, double dtFrameSeconds, double frameStartTime,
    int maxEvents, uint8_t *outputBuffer)
{
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const bool inside = (x0 < width && y < height); // NO early return: warp shuffles below
    const int idx0 = y * width + x0;                // multiple of 4 when inside

    float newVal[4] = {0.f, 0.f, 0.f, 0.f};
    float totalChange[4] = {0.f, 0.f, 0.f, 0.f};
    float ratio[4] = {0.f, 0.f, 0.f, 0.f};
    int crossings[4] = {0, 0, 0, 0};
    int polarity[4] = {0, 0, 0, 0};

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
        const float base[4] = {s01.x, s01.z, s23.x, s23.z};
        const float res[4] = {s01.y, s01.w, s23.y, s23.w};

#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const float deltaLight = newVal[j] - base[j];
            const float avgLight = (newVal[j] + base[j]) * 0.5f;
            const float c = (k1_over_dt_us * deltaLight) / (avgLight + k2) + k4 + k5 * avgLight;
            const float tc = res[j] + c * dtFrameUs;
            totalChange[j] = tc;

            if (tc >= thresholdOn)
            {
                crossings[j] = static_cast<int>(floorf(tc / thresholdOn));
                polarity[j] = +1;
                ratio[j] = thresholdOn / tc;
            }
            else if (tc <= -thresholdOff)
            {
                crossings[j] = static_cast<int>(floorf(-tc / thresholdOff));
                polarity[j] = -1;
                ratio[j] = thresholdOff / (-tc);
            }
        }
    }

    // ---- warp-aggregated slot allocation: each lane contributes 0..4 pixels' crossings
    const unsigned full = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int threadTotal = crossings[0] + crossings[1] + crossings[2] + crossings[3];

    int incl = threadTotal; // inclusive prefix sum over the 32 lanes
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
        warpBase = atomicAdd(reinterpret_cast<int *>(outputBuffer), warpTotal);
    warpBase = __shfl_sync(full, warpBase, 0);

    if (threadTotal > 0)
    {
        Event *events = reinterpret_cast<Event *>(outputBuffer + kHeaderBytes);
        int slotBase = warpBase + incl - threadTotal; // exclusive start of this thread's range

#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int n = crossings[j];
            if (n > 0)
            {
                const uint16_t ex = static_cast<uint16_t>(x0 + j);
                const uint16_t ey = static_cast<uint16_t>(y);
                for (int i = 0; i < n; ++i)
                {
                    const int slot = slotBase + i;
                    if (slot >= maxEvents)
                        break;
                    const double eventTime =
                        frameStartTime + static_cast<double>((i + 1) * ratio[j]) * dtFrameSeconds;
                    events[slot] = Event{ex, ey, eventTime, static_cast<int8_t>(polarity[j])};
                }
                slotBase += n; // advance even if we broke out on overflow (keeps ranges disjoint)
                totalChange[j] -= polarity[j] * n * (polarity[j] > 0 ? thresholdOn : thresholdOff);
            }
        }
    }

    // ---- store all 4 states vectorized (baseIntensity always updates, as in the scalar kernel)
    if (inside)
    {
        float4 *sp = reinterpret_cast<float4 *>(states + idx0);
        sp[0] = make_float4(newVal[0], totalChange[0], newVal[1], totalChange[1]);
        sp[1] = make_float4(newVal[2], totalChange[2], newVal[3], totalChange[3]);
    }
}

/* ---------------------------------------------------------------------------
   Host side, in EventGeneratorCUDA::generate(), replacing the fast-path launch:

        const bool vec4 = (width_ % 4 == 0);
        const dim3 block(32, 8);
        const dim3 grid4(((width_ / 4) + block.x - 1) / block.x,
                         (height_ + block.y - 1) / block.y);

        if (config_.fastDeterministicMode)
        {
            if (vec4)
                GenerateEventsKernelDVSFast4<<<grid4, block, 0, stream_>>>(...same args...);
            else
                GenerateEventsKernelDVSFast<<<grid, block, 0, stream_>>>(...same args...);
        }

   Notes:
   * InitFirstFrameKernel and the state layout are unchanged (PixelStateGPU is 8 bytes,
     cudaMalloc gives >=256B alignment, so the float4 accesses are aligned when width%4==0).
   * Event order in the output differs slightly from the scalar kernel (slots are grouped
     per thread, i.e. per 4 pixels), but the set of events and their timestamps are identical.
   * Try block(32, 4) or (32, 16) too: with 4x fewer threads, occupancy/tail effects change.
--------------------------------------------------------------------------- */
