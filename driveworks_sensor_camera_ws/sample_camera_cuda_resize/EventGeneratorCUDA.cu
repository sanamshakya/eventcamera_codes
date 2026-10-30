#include "EventGeneratorCUDA.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include "FloorTest.cuh"

#define CUDA_CHECK(expr)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err_ = (expr);                                           \
        if (err_ != cudaSuccess)                                             \
        {                                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, \
                    __LINE__, cudaGetErrorString(err_));                     \
            throw std::runtime_error("CUDA call failed");                    \
        }                                                                    \
    } while (0)

// Optional per-phase cycle profiling for the stochastic kernels.
// Build with -DEVSIM_KERNEL_PROF to enable (adds atomics: use for relative
// comparison between phases, not absolute timing).
#ifdef EVSIM_KERNEL_PROF
#define PROF_BEGIN() long long prof_t_ = clock64()
#define PROF_LAP(ctr, ph)                                                             \
    do                                                                                \
    {                                                                                 \
        const long long n_ = clock64();                                               \
        atomicAdd(&(ctr)->cycles[ph], static_cast<unsigned long long>(n_ - prof_t_)); \
        prof_t_ = clock64();                                                          \
    } while (0)
#define PROF_ONLY(stmt) stmt
#else
#define PROF_BEGIN() \
    do               \
    {                \
    } while (0)
#define PROF_LAP(ctr, ph) \
    do                    \
    {                     \
    } while (0)
#define PROF_ONLY(stmt) \
    do                  \
    {                   \
    } while (0)
#endif

namespace evsim
{

    static constexpr size_t kHeaderBytes =
        ((sizeof(int) + alignof(Event) - 1) / alignof(Event)) * alignof(Event);

    namespace
    {

        // Initializes the first frame. Frame timing lives host-side.
        __global__ void InitFirstFrameKernel(PixelStateGPU *states,
                                             const uint16_t *image,
                                             int width, int height, int stride)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= width || y >= height)
                return;

            float intensity = static_cast<float>(image[static_cast<size_t>(y) * stride + x]);
            states[y * width + x] = PixelStateGPU{intensity, 0.0f};
        }

        // Device-input path: copies a pitch-linear uint16 device image (pitch in
        // ELEMENTS) into the tightly packed d_image_ buffer, applying a right
        // shift (e.g. 4 for 12-bit data left-aligned in 16 bits).
        __global__ void RepackShiftKernel(const uint16_t *src, size_t pitchElems,
                                          uint16_t *dst, int w, int h, int shift)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= w || y >= h)
                return;
            dst[static_cast<size_t>(y) * w + x] =
                static_cast<uint16_t>(src[static_cast<size_t>(y) * pitchElems + x] >> shift);
        }

        //======================================================================
        // Deterministic fast path -- mirrors dvs::EventSim::set_deterministic_mode()
        // (event_sim.cpp). No diffusion, no RNG: the drift is a straight-line ramp
        // over the frame, floor-divided by the thresholds. Unchanged.
        //======================================================================
        __global__ void GenerateEventsKernelDVSFast(
            PixelStateGPU *states, const uint16_t *image,
            int width, int height, int stride,
            float k1_over_dt_us, float k2, float k4, float k5,
            float thresholdOn, float thresholdOff,
            float dtFrameUs, double dtFrameSeconds, double frameStartTime,
            int maxEvents, uint8_t *outputBuffer)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            const bool inside = (x < width && y < height); // NO early return: all lanes must hit the shuffles

            const int idx = y * width + x;
            float newVal = 0.f, totalChange = 0.f;
            int crossings = 0;
            int polarity = 0;
            float ratio = 0.f; // step / |totalChange|

            if (inside)
            {
                const PixelStateGPU state = states[idx];
                newVal = static_cast<float>(image[static_cast<size_t>(y) * stride + x]);
                const float baseVal = state.baseIntensity;
                const float deltaLight = newVal - baseVal;
                const float avgLight = (newVal + baseVal) * 0.5f;
                const float c = (k1_over_dt_us * deltaLight) / (avgLight + k2) + k4 + k5 * avgLight;
                totalChange = state.deltaVdRes + c * dtFrameUs;

                if (totalChange >= thresholdOn)
                {
                    crossings = static_cast<int>(floorf(totalChange / thresholdOn));
                    polarity = +1;
                    ratio = thresholdOn / totalChange;
                }
                else if (totalChange <= -thresholdOff)
                {
                    crossings = static_cast<int>(floorf(-totalChange / thresholdOff));
                    polarity = -1;
                    ratio = thresholdOff / (-totalChange);
                }
            }

            // ---- warp-aggregated slot allocation (blockDim.x == 32, so a warp is one row segment)
            const unsigned full = 0xffffffffu;
            const int lane = (threadIdx.y * blockDim.x + threadIdx.x) & 31;
            int incl = crossings;
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

            if (crossings > 0)
            {
                Event *events = reinterpret_cast<Event *>(outputBuffer + kHeaderBytes);
                const int base = warpBase + incl - crossings;
                for (int i = 0; i < crossings; ++i)
                {
                    const int slot = base + i;
                    if (slot >= maxEvents)
                        break;
                    const double eventTime =
                        frameStartTime + static_cast<double>((i + 1) * ratio) * dtFrameSeconds;
                    events[slot] = Event{static_cast<uint16_t>(x), static_cast<uint16_t>(y),
                                         eventTime, static_cast<int8_t>(polarity)};
                }
                totalChange -= polarity * crossings * (polarity > 0 ? thresholdOn : thresholdOff);
            }

            if (inside)
            {
                PixelStateGPU s;
                s.baseIntensity = newVal;
                s.deltaVdRes = totalChange;
                states[idx] = s;
            }
        }

        //======================================================================
        // Full stochastic path -- Bernoulli polarity trial + Inverse Gaussian /
        // Levy waiting time, mirroring the CPU while-loop in event_sim.cpp.
        //
        // Differences from the previous GPU version:
        //   * float, not double
        //   * stateless Philox RNG (no curandState buffer, no RNG init kernel)
        //   * cancellation-free IG / tail-accurate truncated-normal & Levy
        //   * two passes: a cheap per-pixel GATE selects candidate pixels, then
        //     the expensive event kernel runs on the candidate list only
        //   * warp-aggregated event slot allocation
        //======================================================================
        namespace stochastic_device
        {
            using Rng = curandStatePhilox4_32_10_t;

            // 128 draws reserved per pixel per frame (cap 16 iters * <=5 draws = 80).
            __device__ __forceinline__ void init_rng(Rng *r, unsigned long long seed,
                                                     int idx, unsigned long long frame)
            {
                curand_init(seed, static_cast<unsigned long long>(idx), frame * 128ull, r);
            }

            __device__ __forceinline__ float first_on_probability(float epOnReal, float epOffReal,
                                                                  float c, float sigma)
            {
                if (c == 0.f)
                    return 0.5f;
                const float invS2 = 1.f / (sigma * sigma);
                // float exp overflows near 88 -> clamp at 80 (the double path clamped at 700)
                const float inB = fminf(fmaxf(2.f * c * epOffReal * invS2, -80.f), 80.f);
                const float inA = fminf(fmaxf(-2.f * c * epOnReal * invS2, -80.f), 80.f);
                const float eB = expf(inB), eA = expf(inA);
                float den = eB - eA;
                if (fabsf(den) < 1e-8f)
                    den = 1e-8f;
                const float p = (eB - 1.f) / den;
                if (isnan(p))
                    return 0.5f;
                if (isinf(p))
                    return p > 0.f ? 1.f : 0.f;
                return fminf(fmaxf(p, 0.f), 1.f);
            }

            // N(0,1) truncated to (-inf, b]. Phi^-1(p) = -sqrt2 * erfcinv(2p) stays
            // accurate in the lower tail, where erfinv(2p-1) collapses in float.
            __device__ __forceinline__ float sample_trunc_normal_upper(float b, Rng *rng)
            {
                const float u = curand_uniform(rng); // (0,1]
                const float betaCdf = 0.5f * erfcf(-b * 0.70710678f);
                const float p = betaCdf * u;
                const float x = -1.41421356f * erfcinvf(2.f * p);
                return fminf(x, b);
            }

            // erfinv(1-u) == erfcinv(u). u==1 gives inf, caught by the deltaT guard.
            __device__ __forceinline__ float sample_levy(float cScale, Rng *rng)
            {
                const float u = curand_uniform(rng);
                const float ev = 1.41421356f * erfcinvf(u);
                return cScale / (ev * ev);
            }

            __device__ __forceinline__ float sample_non_c_zero(float ep, float c, float sigma, Rng *rng)
            {
                float X;
                if (c > 0.f)
                    X = curand_normal(rng);
                else
                {
                    const float xMax = -sqrtf(-4.f * ep * c / (sigma * sigma));
                    X = sample_trunc_normal_upper(xMax, rng);
                }
                const float mean = ep / c;
                const float scale = (ep / sigma) * (ep / sigma); // lambda
                const float Y = mean * X * X;
                const float Z = 4.f * scale * Y + Y * Y;
                // Original: X2 = mean + mu*(Y - sqrt(Z)), mu = mean/(2*scale).
                // (Y - sqrtZ)(Y + sqrtZ) = -4*scale*Y  =>  X2 = 4*scale*mean*Y/(Y+sqrtZ)^2
                // Same value in exact arithmetic, without the catastrophic cancellation.
                const float den = Y + sqrtf(Z);
                const float X2 = (den != 0.f) ? (4.f * scale * mean * Y) / (den * den) : mean;

                const float U = curand_uniform(rng);
                return (U > mean / (mean + X2)) ? (mean * mean / X2) : X2;
            }

            __device__ __forceinline__ float sample_IG(float ep, float c, float sigma, Rng *rng)
            {
                if (c == 0.f)
                    return sample_levy((ep / sigma) * (ep / sigma), rng);
                return sample_non_c_zero(ep, c, sigma, rng);
            }

            // Drift c and diffusion sigma for one pixel (shared by both kernels).
            __device__ __forceinline__ void drift_diffusion(float newVal, float baseVal,
                                                            float k1_over_dt_us, float k2, float k3,
                                                            float k4, float k5, float k6,
                                                            float &c, float &sigma)
            {
                const float avg = 0.5f * (newVal + baseVal);
                const float inv = 1.f / (avg + k2);
                c = k1_over_dt_us * (newVal - baseVal) * inv + k4 + k5 * avg;
                sigma = k3 * sqrtf(fmaxf(avg, 0.f)) * inv + k6;
            }
        } // namespace stochastic_device

        //----------------------------------------------------------------------
        // PASS A -- gate. Launch with block (32, N): one warp == one row segment.
        //
        // A pixel is skipped when neither threshold can plausibly be reached in
        // this frame (m = c*dt, S = sigma*sqrt(dt)):
        //     up   : max(m,0)  + z*S  <  epOn  - res
        //     down : max(-m,0) + z*S  <  epOff + res
        // The max of a drifted Brownian path is <= drift part + z*S except with
        // probability ~2Q(z) per barrier: z=4 -> 6.3e-5, z=5 -> 5.7e-7, z=6 -> 2e-9.
        // The gate includes the leak/noise terms (k4,k5,k6), so a static scene is
        // NOT assumed silent. Skipped pixels advance their state with one Gaussian
        // step, clamped just inside the thresholds.
        //----------------------------------------------------------------------
        __global__ void StochGateKernel(
            PixelStateGPU *states, const uint16_t *image,
            int width, int height, int stride,
            float k1_over_dt_us, float k2, float k3, float k4, float k5, float k6,
            float thresholdOn, float thresholdOff, float dtFrameUs,
            float gateZ, bool gateEnabled,
            unsigned long long seed, unsigned long long frame,
            int *candList, StochCounters *ctr)
        {
            using namespace stochastic_device;
            PROF_BEGIN();

            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            const bool inside = (x < width && y < height); // NO early return: warp ballots below
            const int idx = y * width + x;
            const unsigned full = 0xffffffffu;
            const int lane = threadIdx.x & 31;

            bool cand = false;
            PixelStateGPU st{0.f, 0.f};
            float newVal = 0.f, m = 0.f, sDt = 0.f;

            if (inside)
            {
                st = states[idx];
                newVal = static_cast<float>(image[static_cast<size_t>(y) * stride + x]);
                PROF_LAP(ctr, PH_G_LOAD);

                float c, sigma;
                drift_diffusion(newVal, st.baseIntensity, k1_over_dt_us, k2, k3, k4, k5, k6, c, sigma);
                m = c * dtFrameUs;
                sDt = sigma * sqrtf(dtFrameUs);
                PROF_LAP(ctr, PH_G_DRIFT);

                const float reachUp = fmaxf(m, 0.f) + gateZ * sDt;
                const float reachDn = fmaxf(-m, 0.f) + gateZ * sDt;
                cand = !gateEnabled ||
                       reachUp >= thresholdOn - st.deltaVdRes ||
                       reachDn >= thresholdOff + st.deltaVdRes ||
                       !isfinite(reachUp + reachDn); // NaN -> full path (it has the guard)
                PROF_LAP(ctr, PH_G_TEST);
            }

            // warp-aggregated append to the candidate list
            const unsigned mask = __ballot_sync(full, cand);
            if (mask)
            {
                const int leader = __ffs(mask) - 1;
                int base = 0;
                if (lane == leader)
                    base = static_cast<int>(atomicAdd(&ctr->numCandidates, static_cast<unsigned>(__popc(mask))));
                base = __shfl_sync(full, base, leader);
                if (cand)
                    candList[base + __popc(mask & ((1u << lane) - 1u))] = idx;
            }
            PROF_LAP(ctr, PH_G_APPEND);

            if (inside && !cand)
            {
                Rng rng;
                init_rng(&rng, seed, idx, frame);
                float r = st.deltaVdRes + m + sDt * curand_normal(&rng);
                r = fminf(fmaxf(r, -0.999f * thresholdOff), 0.999f * thresholdOn);
                states[idx] = PixelStateGPU{newVal, r};
                PROF_LAP(ctr, PH_G_ADVANCE);
            }
        }

#include "StochGateKernel4.inc"

        //----------------------------------------------------------------------
        // PASS B -- full stochastic simulation, candidates only. 1-D launch
        // (blockDim.x must be a multiple of 32), grid-stride over the device-side
        // candidate count. All lanes of a warp stay in the iteration loop until
        // every lane is done so the warp-aggregated atomics are legal.
        //----------------------------------------------------------------------
        __global__ void StochEventKernel(
            PixelStateGPU *states, const uint16_t *image, const int *candList,
            int width, int stride,
            float k1_over_dt_us, float k2, float k3, float k4, float k5, float k6,
            float thresholdOn, float thresholdOff,
            float dtFrameUs, double dtFrameSeconds, double frameStartTime,
            unsigned long long seed, unsigned long long frame,
            int maxEvents, uint8_t *outputBuffer, StochCounters *ctr)
        {
            using namespace stochastic_device;

            const unsigned full = 0xffffffffu;
            const int lane = threadIdx.x & 31;
            const unsigned ltMask = (1u << lane) - 1u;
            const int nCand = static_cast<int>(ctr->numCandidates);
            int *eventCount = reinterpret_cast<int *>(outputBuffer);
            Event *events = reinterpret_cast<Event *>(outputBuffer + kHeaderBytes);
            const float endT = dtFrameUs;

            for (int blockBase = blockIdx.x * blockDim.x; blockBase < nCand;
                 blockBase += gridDim.x * blockDim.x) // uniform per block -> per warp
            {
                PROF_BEGIN();
                const int i = blockBase + threadIdx.x;
                const bool valid = i < nCand;

                int idx = 0;
                float newVal = 0.f, res = 0.f, c = 0.f, sigma = 1.f, startT = 0.f;
                Rng rng;
                if (valid)
                {
                    idx = candList[i];
                    const int px = idx % width, py = idx / width;
                    const PixelStateGPU st = states[idx];
                    newVal = static_cast<float>(image[static_cast<size_t>(py) * stride + px]);
                    res = st.deltaVdRes;
                    drift_diffusion(newVal, st.baseIntensity, k1_over_dt_us, k2, k3, k4, k5, k6, c, sigma);
                    init_rng(&rng, seed, idx, frame);
                }
                PROF_LAP(ctr, PH_B_LOAD);

                bool active = valid;
                int iter = 0;
                for (; iter < kStochMaxCrossings && __any_sync(full, active); ++iter)
                {
                    bool emit = false, on = false;
                    float tEnd = 0.f;

                    if (active)
                    {
                        const float epOnReal = thresholdOn - res;
                        const float epOffReal = thresholdOff + res;
                        const float pOn = first_on_probability(epOnReal, epOffReal, c, sigma);
                        on = curand_uniform(&rng) <= pOn; // Bernoulli trial
                        PROF_LAP(ctr, PH_B_PROB);

                        const float epIn = on ? epOnReal : epOffReal;
                        const float deltaT = sample_IG(epIn, on ? c : -c, sigma, &rng); // IG or Levy
                        PROF_LAP(ctr, PH_B_SAMPLE);

                        // Defensive guard: NaN / non-positive waiting time -> nothing more
                        // resolvable this frame (a NaN would otherwise poison tEnd and the
                        // `tEnd >= endT` comparison is false for every endT).
                        if (!isfinite(deltaT) || deltaT <= 0.f)
                        {
                            active = false;
                            atomicAdd(&ctr->numBadDeltaT, 1u);
                            PROF_ONLY(atomicAdd(&ctr->iterHist[iter], 1u));
                        }
                        else
                        {
                            tEnd = deltaT + startT;
                            if (tEnd >= endT)
                            {
                                res += (on ? 1.f : -1.f) * epIn * (endT - startT) / deltaT;
                                active = false;
                                PROF_ONLY(atomicAdd(&ctr->iterHist[iter], 1u));
                            }
                            else
                                emit = true;
                        }
                    }

                    // warp-aggregated slot allocation for this iteration's events
                    const unsigned em = __ballot_sync(full, emit);
                    if (em)
                    {
                        const int leader = __ffs(em) - 1;
                        int base = 0;
                        if (lane == leader)
                            base = atomicAdd(eventCount, __popc(em));
                        base = __shfl_sync(full, base, leader);
                        if (emit)
                        {
                            const int slot = base + __popc(em & ltMask);
                            if (slot < maxEvents)
                            {
                                const double eventTime = frameStartTime +
                                                         static_cast<double>(tEnd / endT) * dtFrameSeconds;
                                events[slot] = Event{static_cast<uint16_t>(idx % width),
                                                     static_cast<uint16_t>(idx / width),
                                                     eventTime, static_cast<int8_t>(on ? 1 : -1)};
                            }
                            startT = tEnd;
                            res = 0.f;
                        }
                    }
                    PROF_LAP(ctr, PH_B_EMIT);
                }

                if (active) // fell out by exhausting the cap
                {
                    atomicAdd(&ctr->numCapHit, 1u);
                    PROF_ONLY(atomicAdd(&ctr->iterHist[kStochMaxCrossings], 1u));
                }
                if (valid)
                {
                    if (!isfinite(res)) // never let a NaN residual poison the pixel permanently
                        res = 0.f;
                    states[idx] = PixelStateGPU{newVal, res};
                }
                PROF_LAP(ctr, PH_B_STORE);
            }
        }

    } // anonymous namespace

    EventGeneratorCUDA::EventGeneratorCUDA(const Config &config, int maxEventsPerFrame)
        : config_(config), k_(dvs::sensor_k_preset(config.sensorType)), maxEvents_(maxEventsPerFrame)
    {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&numSMs_, cudaDevAttrMultiProcessorCount, dev));
        for (auto &e : sEv_)
            CUDA_CHECK(cudaEventCreate(&e));

        const char *fm = std::getenv("EVSIM_FLOOR_MODE");
        floorMode_ = fm ? std::atoi(fm) : -1;
        fprintf(stderr, "[gen] floorMode=%d\n", floorMode_);
    }

    EventGeneratorCUDA::~EventGeneratorCUDA()
    {
        freeDeviceBuffers();
        for (auto &e : sEv_)
            if (e)
                cudaEventDestroy(e);
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    // Device memory buffer handling

    void EventGeneratorCUDA::freeDeviceBuffers()
    {
        if (d_image_)
            cudaFree(d_image_);
        if (d_pixelStates_)
            cudaFree(d_pixelStates_);
        if (d_outputBuffer_)
            cudaFree(d_outputBuffer_);
        if (d_candList_)
            cudaFree(d_candList_);
        if (d_counters_)
            cudaFree(d_counters_);
        if (h_counters_)
            cudaFreeHost(h_counters_);
        if (h_image_)
            cudaFreeHost(h_image_);
        if (h_outputBuffer_)
            cudaFreeHost(h_outputBuffer_);
        d_image_ = nullptr;
        d_pixelStates_ = nullptr;
        d_outputBuffer_ = nullptr;
        d_candList_ = nullptr;
        d_counters_ = nullptr;
        h_counters_ = nullptr;
        h_image_ = nullptr;
        h_outputBuffer_ = nullptr;
    }

    void EventGeneratorCUDA::allocateDeviceBuffers()
    {
        const size_t numPixels = static_cast<size_t>(width_) * height_;
        outputBufferBytes_ = kHeaderBytes + static_cast<size_t>(maxEvents_) * sizeof(Event);

        CUDA_CHECK(cudaMalloc(&d_image_, numPixels * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_pixelStates_, numPixels * sizeof(PixelStateGPU)));
        CUDA_CHECK(cudaMalloc(&d_outputBuffer_, outputBufferBytes_));

        // Stochastic path only: candidate list + counters (no RNG state buffer needed).
        if (!config_.fastDeterministicMode)
        {
            CUDA_CHECK(cudaMalloc(&d_candList_, numPixels * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_counters_, sizeof(StochCounters)));
            CUDA_CHECK(cudaHostAlloc(&h_counters_, sizeof(StochCounters), cudaHostAllocDefault));
        }

        // Pinned (page-locked) host memory for predictable DMA timing.
        CUDA_CHECK(cudaHostAlloc(&h_image_, numPixels * sizeof(uint16_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_outputBuffer_, outputBufferBytes_, cudaHostAllocDefault));
    }

    void EventGeneratorCUDA::initialize(int width, int height)
    {
        freeDeviceBuffers();
        width_ = width;
        height_ = height;
        allocateDeviceBuffers();
        initialized_ = false;
    }

    void EventGeneratorCUDA::reset()
    {
        initialized_ = false;
    }

    // Accumulates per-frame stochastic stats (counters were copied back on the
    // existing sync) and prints a report every 100 frames.
    void EventGeneratorCUDA::accumulateStochProfile(size_t numPixels)
    {
        if (frameNumber_ <= 5) // warm-up
            return;
        float msG = 0.f, msE = 0.f;
        cudaEventElapsedTime(&msG, sEv_[0], sEv_[1]);
        cudaEventElapsedTime(&msE, sEv_[1], sEv_[2]);
        auto &p = sprof_;
        p.frames++;
        p.msGate += msG;
        p.msEvents += msE;
        p.cand += h_counters_->numCandidates;
        p.capHit += h_counters_->numCapHit;
        p.bad += h_counters_->numBadDeltaT;
        for (int i = 0; i <= kStochMaxCrossings; ++i)
            p.hist[i] += h_counters_->iterHist[i];
        for (int i = 0; i < PH_COUNT; ++i)
            p.cyc[i] += h_counters_->cycles[i];

        if (p.frames < 100)
            return;
        const double nF = p.frames;
        fprintf(stderr,
                "[stoch] gate %.3f ms | events %.3f ms | candidates %.2f%% of pixels (%.0f/frame) "
                "| capHit %.2f/frame | badDeltaT %.2f/frame\n",
                p.msGate / nF, p.msEvents / nF, 100.0 * p.cand / (nF * numPixels), p.cand / nF,
                p.capHit / nF, p.bad / nF);
#ifdef EVSIM_KERNEL_PROF
        static const char *names[PH_COUNT] = {
            "G.load", "G.drift", "G.test", "G.append", "G.advance",
            "B.load+rng", "B.pOn", "B.sampleIG", "B.emit(atomic)", "B.store"};
        for (int i = 0; i < PH_COUNT; ++i)
        {
            const double denom = (i < PH_B_LOAD) ? nF * numPixels : double(p.cand);
            fprintf(stderr, "[stoch]   %-16s %8.1f cycles/thread\n", names[i], p.cyc[i] / denom);
        }
        fprintf(stderr, "[stoch]   events/pixel histogram (candidates):");
        for (int i = 0; i <= kStochMaxCrossings; ++i)
            fprintf(stderr, " %d:%llu", i, p.hist[i]);
        fprintf(stderr, "\n");
#endif
        p = StochProfHost{};
    }

    // ------------------------------------------------------------------------
    // Host-input entry point: image is a HOST pointer, stride is in PIXELS.
    // ------------------------------------------------------------------------
    EventPacket EventGeneratorCUDA::generate(
        const uint16_t *image,
        int width,
        int height,
        int stridePixels,
        double timestamp)
    {
        EventPacket packet{};
        if (image == nullptr)
            return packet;
        const double tFn0 = nowMs(); // [prof] start of generate()

        // Lazy init safety net (does NOT handle a resolution change mid-stream).
        if (width_ == 0 || height_ == 0)
            initialize(width, height);

        if (width_ == 0 || height_ == 0)
            return packet;

        // Stage into pinned memory row-by-row (handles stridePixels != width_).
        const double tS0 = nowMs(); // [prof]
        for (int y = 0; y < height_; ++y)
        {
            std::memcpy(h_image_ + static_cast<size_t>(y) * width_,
                        image + static_cast<size_t>(y) * stridePixels,
                        static_cast<size_t>(width_) * sizeof(uint16_t));
        }
        prof_.hostStage = nowMs() - tS0; // [prof]

        const size_t h2dBytes = static_cast<size_t>(width_) * height_ * sizeof(uint16_t);
        cudaEventRecord(prof_.ev[0], stream_); // [prof] before H2D
        CUDA_CHECK(cudaMemcpyAsync(d_image_, h_image_, h2dBytes,
                                   cudaMemcpyHostToDevice, stream_));
        cudaEventRecord(prof_.ev[1], stream_); // [prof] after H2D

        return runFrame(timestamp, tFn0);
    }

    // ------------------------------------------------------------------------
    // Device-input entry point: dImage is a DEVICE pointer, pitch is in BYTES.
    // Copies GPU->GPU into the packed d_image_ buffer, applying `rawShift`.
    // ------------------------------------------------------------------------
    EventPacket EventGeneratorCUDA::generateFromDevice(
        const uint16_t *dImage,
        int width,
        int height,
        size_t pitchBytes,
        double timestamp,
        int rawShift)
    {
        EventPacket packet{};
        if (dImage == nullptr)
            return packet;
        const double tFn0 = nowMs(); // [prof]

        // Lazy init safety net (does NOT handle a resolution change mid-stream).
        if (width_ == 0 || height_ == 0)
            initialize(width, height);

        if (width_ == 0 || height_ == 0)
            return packet;

        if (width != width_ || height != height_)
        {
            fprintf(stderr, "generateFromDevice: size %dx%d != generator %dx%d, frame skipped\n",
                    width, height, width_, height_);
            return packet;
        }
        if (pitchBytes < static_cast<size_t>(width) * sizeof(uint16_t) ||
            (pitchBytes % sizeof(uint16_t)) != 0)
        {
            throw std::runtime_error("generateFromDevice: invalid pitch (must be bytes, >= width*2, even)");
        }

        prof_.hostStage = 0.0; // no host staging on this path

        // The producer (camera/ISP) may have written this buffer on another
        // stream/engine; make sure it is finished before our stream reads it.
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(prof_.ev[0], stream_); // [prof] before repack
        const dim3 blk(32, 8);
        const dim3 grd((width_ + blk.x - 1) / blk.x, (height_ + blk.y - 1) / blk.y);
        RepackShiftKernel<<<grd, blk, 0, stream_>>>(dImage, pitchBytes / sizeof(uint16_t),
                                                    d_image_, width_, height_, rawShift);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(prof_.ev[1], stream_); // [prof] after repack

        // Debug only (env EVSIM_INPUT_STATS=1): intensity range the kernels really see
        // (after the shift) and the frame interval. The sensor K presets are
        // calibrated for roughly 0..255 intensities; a raw range of 0..4095 or
        // 0..65535 inflates the leak term k5*avg*dt and with it the event count.
        static const bool inputStats = std::getenv("EVSIM_INPUT_STATS") != nullptr;
        if (inputStats && (static_cast<unsigned long long>(frameNumber_) % 30ull) == 0ull)
        {
            const size_t n = static_cast<size_t>(width_) * height_;
            CUDA_CHECK(cudaMemcpyAsync(h_image_, d_image_, n * sizeof(uint16_t),
                                       cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            uint16_t mn = 0xFFFF, mx = 0;
            double sum = 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                const uint16_t v = h_image_[i];
                mn = std::min(mn, v);
                mx = std::max(mx, v);
                sum += v;
            }
            fprintf(stderr, "[input] frame %llu: shift %d -> min %u max %u mean %.1f | dt %.3f ms\n",
                    static_cast<unsigned long long>(frameNumber_), rawShift, mn, mx, sum / n,
                    initialized_ ? (timestamp - previousTimestamp_) * 1e3 : 0.0);
        }

        return runFrame(timestamp, tFn0);
    }

    // ------------------------------------------------------------------------
    // Shared simulation tail. Precondition: d_image_ holds the current frame
    // (packed, stride == width_) and the copy/repack is already enqueued on
    // stream_.
    // ------------------------------------------------------------------------
    EventPacket EventGeneratorCUDA::runFrame(double timestamp, double tFn0)
    {
        EventPacket packet{};
        const size_t h2dBytes = static_cast<size_t>(width_) * height_ * sizeof(uint16_t);

        const int cuda_blocks = 32;
        const int cuda_threads = 8;
        const dim3 block(cuda_blocks, cuda_threads);
        const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);

        if (!initialized_)
        {
            // The Philox RNG is stateless, so there is no RNG init kernel anymore.
            InitFirstFrameKernel<<<grid, block, 0, stream_>>>(
                d_pixelStates_, d_image_, width_, height_, width_);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream_));

            initialized_ = true;
            previousTimestamp_ = timestamp;
            packet.startTime = timestamp;
            packet.endTime = timestamp;
            packet.frameNumber = frameNumber_++;
            return packet;
        }

        // Guard against a non-increasing timestamp (jitter/repeats from real clocks).
        double dtFrameSeconds = timestamp - previousTimestamp_;
        if (dtFrameSeconds <= 0.0)
        {
            dtFrameSeconds = 1e-6;
        }
        // Sensor K coefficients are calibrated for timestamps in MICROSECONDS.
        // Event timestamps themselves are still produced in seconds.
        const double dtFrameUs = dtFrameSeconds * 1e6;

        // Reset only the leading event-count int, not the whole output buffer.
        CUDA_CHECK(cudaMemsetAsync(d_outputBuffer_, 0, sizeof(int), stream_));

        if (floorMode_ >= 0)
        {
            // Diagnostic: move a controlled number of bytes per pixel, nothing else.
            // Mirrors the stochastic branch's counter reset and sEv_ events, so the
            // [stoch] gate/events timing code further down keeps working unchanged.
            if (d_counters_)
                CUDA_CHECK(cudaMemsetAsync(d_counters_, 0, sizeof(StochCounters), stream_));
            cudaEventRecord(sEv_[0], stream_);

            launchFloorTest(floorMode_, d_image_, d_pixelStates_,
                            width_, height_, width_, stream_);

            cudaEventRecord(sEv_[1], stream_);
            cudaEventRecord(sEv_[2], stream_); // no pass B, so the "events" time reads ~0
            if (d_counters_)
                CUDA_CHECK(cudaMemcpyAsync(h_counters_, d_counters_, sizeof(StochCounters),
                                           cudaMemcpyDeviceToHost, stream_));
        }
        else if (config_.fastDeterministicMode)
        {
            GenerateEventsKernelDVSFast<<<grid, block, 0, stream_>>>(
                d_pixelStates_, d_image_, width_, height_, width_,
                static_cast<float>(k_.k1 / dtFrameUs),
                static_cast<float>(k_.k2), static_cast<float>(k_.k4), static_cast<float>(k_.k5),
                static_cast<float>(config_.contrastThresholdOn),
                static_cast<float>(config_.contrastThresholdOff),
                static_cast<float>(dtFrameUs), dtFrameSeconds, previousTimestamp_,
                maxEvents_, d_outputBuffer_);
        }
        else
        {
            const size_t numPixels = static_cast<size_t>(width_) * height_;
            CUDA_CHECK(cudaMemsetAsync(d_counters_, 0, sizeof(StochCounters), stream_));
            cudaEventRecord(sEv_[0], stream_);

            const bool vec4 = (width_ % 4 == 0); // stride == width_ in this code path
            if (vec4)
            {
                const dim3 block4(32, 8); // blockDim.x must stay 32
                const dim3 grid4(((width_ >> 2) + block4.x - 1) / block4.x,
                                 (height_ + block4.y - 1) / block4.y); // 1280x720 -> (10, 90)
                StochGateKernel4<<<grid4, block4, 0, stream_>>>(
                    d_pixelStates_, d_image_, width_, height_, width_,
                    static_cast<float>(k_.k1 / dtFrameUs), static_cast<float>(k_.k2), static_cast<float>(k_.k3),
                    static_cast<float>(k_.k4), static_cast<float>(k_.k5), static_cast<float>(k_.k6),
                    static_cast<float>(config_.contrastThresholdOn),
                    static_cast<float>(config_.contrastThresholdOff),
                    static_cast<float>(dtFrameUs), gateZ_, gateEnabled_,
                    static_cast<unsigned long long>(rngSeed_), static_cast<unsigned long long>(frameNumber_),
                    d_candList_, d_counters_);
            }
            else
            {
                // Pass A: per-pixel gate (all pixels, cheap) -> candidate list
                StochGateKernel<<<grid, block, 0, stream_>>>(
                    d_pixelStates_, d_image_, width_, height_, width_,
                    static_cast<float>(k_.k1 / dtFrameUs), static_cast<float>(k_.k2), static_cast<float>(k_.k3),
                    static_cast<float>(k_.k4), static_cast<float>(k_.k5), static_cast<float>(k_.k6),
                    static_cast<float>(config_.contrastThresholdOn),
                    static_cast<float>(config_.contrastThresholdOff),
                    static_cast<float>(dtFrameUs), gateZ_, gateEnabled_,
                    static_cast<unsigned long long>(rngSeed_), static_cast<unsigned long long>(frameNumber_),
                    d_candList_, d_counters_);
            }

            cudaEventRecord(sEv_[1], stream_);

            // Pass B: full stochastic simulation on candidates only
            const int evThreads = 128; // multiple of 32 (required by the warp-level primitives)
            const int evBlocks = std::max(1, std::min<int>(static_cast<int>((numPixels + evThreads - 1) / evThreads),
                                                           numSMs_ * 16));
            StochEventKernel<<<evBlocks, evThreads, 0, stream_>>>(
                d_pixelStates_, d_image_, d_candList_, width_, width_,
                static_cast<float>(k_.k1 / dtFrameUs), static_cast<float>(k_.k2), static_cast<float>(k_.k3),
                static_cast<float>(k_.k4), static_cast<float>(k_.k5), static_cast<float>(k_.k6),
                static_cast<float>(config_.contrastThresholdOn),
                static_cast<float>(config_.contrastThresholdOff),
                static_cast<float>(dtFrameUs), dtFrameSeconds, previousTimestamp_,
                static_cast<unsigned long long>(rngSeed_), static_cast<unsigned long long>(frameNumber_),
                maxEvents_, d_outputBuffer_, d_counters_);
            cudaEventRecord(sEv_[2], stream_);
            // Rides on the existing sync below.
            CUDA_CHECK(cudaMemcpyAsync(h_counters_, d_counters_, sizeof(StochCounters),
                                       cudaMemcpyDeviceToHost, stream_));
        }
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(prof_.ev[2], stream_); // [prof] after kernel

        // Step 1: just the 4-byte event count (event-count-sized copy keeps D2H small)
        CUDA_CHECK(cudaMemcpyAsync(h_outputBuffer_, d_outputBuffer_, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream_));
        const double tY0 = nowMs();
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        int eventCount = *reinterpret_cast<const int *>(h_outputBuffer_);
        if (eventCount > maxEvents_)
        {
            fprintf(stderr, "EventGeneratorCUDA: %d events, capacity %d - %d dropped.\n",
                    eventCount, maxEvents_, eventCount - maxEvents_);
            eventCount = maxEvents_;
        }

        // Step 2: only the events actually produced
        const size_t d2hBytes = sizeof(int) + static_cast<size_t>(eventCount) * sizeof(Event);
        if (eventCount > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(h_outputBuffer_ + kHeaderBytes,
                                       d_outputBuffer_ + kHeaderBytes,
                                       static_cast<size_t>(eventCount) * sizeof(Event),
                                       cudaMemcpyDeviceToHost, stream_));
        }
        cudaEventRecord(prof_.ev[3], stream_); // after D2H
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        prof_.hostSync = nowMs() - tY0;

        const Event *devEvents = reinterpret_cast<const Event *>(h_outputBuffer_ + kHeaderBytes);

        packet.startTime = previousTimestamp_;
        packet.endTime = timestamp;
        packet.frameNumber = frameNumber_++;
        const double tA0 = nowMs();                              // [prof]
        packet.events.assign(devEvents, devEvents + eventCount); // O(eventCount), not O(W*H)
        prof_.hostAssign = nowMs() - tA0;                        // [prof]

        previousTimestamp_ = timestamp;

        if (!config_.fastDeterministicMode)
            accumulateStochProfile(static_cast<size_t>(width_) * height_);

        prof_.hostTotal = nowMs() - tFn0; // [prof]
        prof_.endFrame(static_cast<size_t>(eventCount), d2hBytes, h2dBytes);
        return packet;
    }

} // namespace evsim
