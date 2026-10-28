#include "EventGeneratorCUDA.cuh"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <cmath>

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

namespace evsim
{

    static constexpr size_t kHeaderBytes =
        ((sizeof(int) + alignof(Event) - 1) / alignof(Event)) * alignof(Event);

    namespace
    {

        // Cuda kernel for Initializing first frame.
        // No timestamp parameter anymore -- frame timing lives host-side in
        // EventGeneratorCUDA::previousTimestamp_ (see the .cuh comment).
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

        // Only needed for the stochastic path. Seeded once here and never
        // re-initialized per frame -- curand_init() is relatively expensive, and
        // each pixel's state needs to persist (advance) across frames the same way
        // delta_vd_res/baseIntensity do.
        __global__ void RNGInitKernel(curandState *states, int width, int height,
                                      unsigned long long seed)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= width || y >= height)
                return;
            int idx = y * width + x;
            curand_init(seed, static_cast<unsigned long long>(idx), 0, &states[idx]);
        }

        //======================================================================
        // Deterministic fast path -- see dvs::EventSim::set_deterministic_mode()
        // (event_sim.cpp) for the CPU/OpenMP reference this mirrors exactly. No
        // diffusion (k3/k6 unused), no RNG: the drift is treated as a straight-line
        // ramp over the frame and floor-divided by the thresholds.
        //======================================================================

        // __global__ void GenerateEventsKernelDVSFast(
        //     PixelStateGPU *states,
        //     const uint16_t *image,
        //     int width, int height, int stride,
        //     float k1_over_dt_us, // k1 / dt_frame_in_microseconds -- see the .cu note on units
        //     float k2, float k4, float k5,
        //     float thresholdOn, float thresholdOff,
        //     float dtFrameUs,       // for the drift accumulation
        //     double dtFrameSeconds, // for producing real-world event timestamps
        //     double frameStartTime, // seconds -- previous frame's timestamp
        //     int maxEvents,
        //     uint8_t *outputBuffer /* [int count][Event...] */)
        // {
        //     int x = blockIdx.x * blockDim.x + threadIdx.x;
        //     int y = blockIdx.y * blockDim.y + threadIdx.y;
        //     if (x >= width || y >= height)
        //         return;

        //     const int idx = y * width + x;
        //     PixelStateGPU state = states[idx];

        //     const float newVal = static_cast<float>(image[static_cast<size_t>(y) * stride + x]);
        //     const float baseVal = state.baseIntensity;

        //     const float deltaLight = newVal - baseVal;
        //     const float avgLight = (newVal + baseVal) * 0.5f;
        //     const float denom = 1.0f / (avgLight + k2);

        //     const float c = (k1_over_dt_us * deltaLight) * denom + k4 + k5 * avgLight; // drift

        //     float totalChange = state.deltaVdRes + c * dtFrameUs;

        //     int *eventCount = reinterpret_cast<int *>(outputBuffer);
        //     Event *events = reinterpret_cast<Event *>(outputBuffer + kHeaderBytes);

        //     if (totalChange >= thresholdOn)
        //     {
        //         const int crossings = static_cast<int>(floorf(totalChange / thresholdOn));
        //         if (crossings > 0)
        //         {
        //             const int base = atomicAdd(eventCount, crossings);
        //             for (int i = 0; i < crossings; ++i)
        //             {
        //                 const int slot = base + i;
        //                 if (slot >= maxEvents)
        //                     break; // dropped -- host logs overflow

        //                 const float frac = static_cast<float>(i + 1) * thresholdOn / totalChange;
        //                 const double eventTime = frameStartTime + static_cast<double>(frac) * dtFrameSeconds;

        //                 events[slot] = Event{static_cast<uint16_t>(x), static_cast<uint16_t>(y),
        //                                      eventTime, +1};
        //             }
        //             totalChange -= crossings * thresholdOn;
        //         }
        //     }
        //     else if (totalChange <= -thresholdOff)
        //     {
        //         const int crossings = static_cast<int>(floorf(-totalChange / thresholdOff));
        //         if (crossings > 0)
        //         {
        //             const int base = atomicAdd(eventCount, crossings);
        //             for (int i = 0; i < crossings; ++i)
        //             {
        //                 const int slot = base + i;
        //                 if (slot >= maxEvents)
        //                     break;

        //                 const float frac = static_cast<float>(i + 1) * thresholdOff / (-totalChange);
        //                 const double eventTime = frameStartTime + static_cast<double>(frac) * dtFrameSeconds;

        //                 events[slot] = Event{static_cast<uint16_t>(x), static_cast<uint16_t>(y),
        //                                      eventTime, -1};
        //             }
        //             totalChange += crossings * thresholdOff;
        //         }
        //     }

        //     // Unlike the old linear model, baseIntensity updates EVERY call, not
        //     // just on a crossing -- the DVS-Voltmeter drift is computed relative to
        //     // the immediately preceding frame, not a long-lived quantized reference.
        //     state.baseIntensity = newVal;
        //     state.deltaVdRes = totalChange;
        //     states[idx] = state;
        // }

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

#include "GenerateEventsKernelDVSFast4.inc"

        //======================================================================
        // Full stochastic path -- Bernoulli polarity trial + Inverse
        // Gaussian/Levy waiting time, exactly mirroring the CPU while-loop in
        // event_sim.cpp. erf()/erfinv() are CUDA math-library device builtins --
        // no custom implementation needed here (unlike the CPU dvs_math.hpp, which
        // has to hand-roll erfinv via Newton's method since <cmath> has no inverse).
        //======================================================================
        namespace stochastic_device
        {

            __device__ inline double first_on_probability(double epOnReal, double epOffReal,
                                                          double c, double sigma)
            {
                if (c == 0.0)
                    return 0.5; // see the CPU-side note: this assumes epOn==epOff
                const double sigmaSq = sigma * sigma;
                double expInB = 2.0 * c * epOffReal / sigmaSq;
                double expInA = -2.0 * c * epOnReal / sigmaSq;
                expInB = fmin(fmax(expInB, -700.0), 700.0);
                expInA = fmin(fmax(expInA, -700.0), 700.0);
                const double exp2uB = exp(expInB);
                const double exp2uA = exp(expInA);
                double numerator = exp2uB - 1.0;
                double denominator = exp2uB - exp2uA;
                if (fabs(denominator) < 1e-8)
                    denominator = 1e-8;
                double p = numerator / denominator;
                if (isnan(p))
                    return 0.5;
                if (isinf(p))
                    return p > 0.0 ? 1.0 : 0.0;
                return fmin(fmax(p, 0.0), 1.0);
            }

            __device__ inline double sample_truncated_normal(double mean, double scale,
                                                             double a, double b,
                                                             curandState *rng)
            {
                const double u = curand_uniform_double(rng);
                const double alphaCdf = isinf(a) ? 0.0 : 0.5 * (1.0 + erf((a - mean) / scale / sqrt(2.0)));
                const double betaCdf = 0.5 * (1.0 + erf((b - mean) / scale / sqrt(2.0)));
                const double p = alphaCdf + (betaCdf - alphaCdf) * u;
                const double v = fmin(fmax(2.0 * p - 1.0, -1.0), 1.0);
                double out = mean + scale * sqrt(2.0) * erfinv(v);
                return fmin(fmax(out, a), b);
            }

            __device__ inline double sample_levy(double cScale, curandState *rng)
            {
                const double u = curand_uniform_double(rng);
                const double ev = sqrt(2.0) * erfinv(1.0 - u);
                return cScale / (ev * ev);
            }

            __device__ inline double sample_non_c_zero(double ep, double c, double sigma,
                                                       curandState *rng)
            {
                double X;
                if (c > 0.0)
                {
                    X = curand_normal_double(rng);
                }
                else
                {
                    const double xMaxThres = -sqrt(-4.0 * ep * c / (sigma * sigma));
                    X = sample_truncated_normal(0.0, 1.0, -INFINITY, xMaxThres, rng);
                }
                const double mean = ep / c;
                const double lambdaIg = (ep / sigma) * (ep / sigma);
                const double scale = lambdaIg;
                const double mu = mean / (2.0 * scale);

                const double Y = mean * X * X;
                const double Z = 4.0 * scale * Y + Y * Y;
                const double X2 = mean + mu * (Y - sqrt(Z));

                const double U = curand_uniform_double(rng);
                return (U > mean / (mean + X2)) ? (mean * mean / X2) : X2;
            }

            __device__ inline double sample_IG(double ep, double c, double sigma, curandState *rng)
            {
                if (c == 0.0)
                {
                    const double scaleLevy = (ep / sigma) * (ep / sigma);
                    return sample_levy(scaleLevy, rng);
                }
                return sample_non_c_zero(ep, c, sigma, rng);
            }

        } // namespace stochastic_device

        // Hard cap on threshold crossings simulated per pixel per frame. This is a
        // SAFETY VALVE, not a tuning knob -- under any sanely-configured
        // threshold/sigma combination, a pixel crossing more than a handful of
        // times in one frame would already indicate misconfigured parameters (see
        // the earlier note on near-zero thresholds / unscaled leak terms). Its real
        // job is bounding worst-case kernel runtime so a NaN or a degenerate
        // parameter combination can never produce a genuinely unbounded loop, which
        // is what was causing the GPU watchdog to kill the kernel (surfacing later
        // as a deferred error on the next cudaMemcpyAsync/cudaStreamSynchronize).
        constexpr int kMaxCrossingsPerPixel = 16;

        __global__ void GenerateEventsKernelDVSStochastic(
            PixelStateGPU *states, curandState *rngStates,
            const uint16_t *image, int width, int height, int stride,
            float k1, float k2, float k3, float k4, float k5, float k6,
            float thresholdOn, float thresholdOff,
            float dtFrameUs, double dtFrameSeconds, double frameStartTime,
            int maxEvents, uint8_t *outputBuffer)
        {
            using namespace stochastic_device;

            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= width || y >= height)
                return;
            const int idx = y * width + x;

            PixelStateGPU state = states[idx];
            curandState rng = rngStates[idx]; // load once, work on a local copy, store back at the end

            const double newVal = static_cast<double>(image[static_cast<size_t>(y) * stride + x]);
            const double baseVal = static_cast<double>(state.baseIntensity);

            const double deltaLight = newVal - baseVal;
            const double avgLight = (newVal + baseVal) * 0.5;
            const double denom = 1.0 / (avgLight + static_cast<double>(k2));

            const double muClean = (static_cast<double>(k1) / static_cast<double>(dtFrameUs)) * deltaLight * denom;
            const double c = muClean + static_cast<double>(k4) + static_cast<double>(k5) * avgLight; // drift
            const double varClean = (static_cast<double>(k3) * sqrt(fmax(avgLight, 0.0))) * denom;
            const double sigma = varClean + static_cast<double>(k6); // diffusion

            const double epOn = thresholdOn;
            const double epOff = thresholdOff;
            double deltaVdLegacy = state.deltaVdRes;

            // Time within this frame is tracked in [0, dtFrameUs] here (relative),
            // rather than absolute microsecond timestamps as on the CPU side --
            // simpler on-device, and frac = tEndIdeal/dtFrameUs is all that's
            // needed to place the event's real-world timestamp below.
            double startT = 0.0;
            const double endT = static_cast<double>(dtFrameUs);

            int *eventCount = reinterpret_cast<int *>(outputBuffer);
            Event *events = reinterpret_cast<Event *>(outputBuffer + kHeaderBytes);

            // Bounded loop, per your suggestion: run at most kMaxCrossingsPerPixel
            // times, but each iteration still checks the same "does this actually
            // fit in the remaining frame time" condition as before (the `if
            // (tEndIdeal >= endT) { ...; break; }` below) -- so in the overwhelming
            // majority of frames this exits after 0-2 iterations exactly as it did
            // with `while(true)`; the cap only ever matters in the pathological
            // cases that used to hang.
            bool exhaustedCap = true; // becomes false if we hit the normal "doesn't fit" exit
            for (int iter = 0; iter < kMaxCrossingsPerPixel; ++iter)
            {
                const double epOnReal = epOn - deltaVdLegacy;
                const double epOffReal = epOff + deltaVdLegacy;

                const double pOn = first_on_probability(epOnReal, epOffReal, c, sigma);
                const bool on = curand_uniform_double(&rng) <= pOn; // Bernoulli trial

                const double epInput = on ? epOnReal : epOffReal;
                const double cInput = on ? c : -c;
                const double deltaT = sample_IG(epInput, cInput, sigma, &rng); // IG or Levy

                // Defensive guard: sample_IG should always return a finite,
                // positive value, but edge-case (ep, c, sigma) combinations
                // (e.g. sigma underflowing to ~0) can in principle produce NaN
                // or <=0. Without this check, a NaN here poisons `tEndIdeal` and
                // `tEndIdeal >= endT` is false for ALL endT (IEEE 754: any
                // comparison with NaN is false) -- the old code's actual root
                // cause of an unconditional hang. Treat it as "nothing more
                // resolvable this frame" and bail out cleanly instead.
                if (!isfinite(deltaT) || deltaT <= 0.0)
                {
                    exhaustedCap = false;
                    break;
                }

                const double tEndIdeal = deltaT + startT;

                if (tEndIdeal >= endT)
                {
                    const double sign = on ? 1.0 : -1.0;
                    deltaVdLegacy += sign * epInput * (endT - startT) / deltaT;
                    exhaustedCap = false;
                    break;
                }

                // Note: unlike the deterministic kernel's single atomicAdd(crossings),
                // this claims one slot per emitted event -- per-pixel event counts are
                // typically 0-3, so the extra atomic contention is minor.
                const int slot = atomicAdd(eventCount, 1);
                if (slot < maxEvents)
                {
                    const double frac = tEndIdeal / endT; // 0..1 fraction of the frame elapsed
                    const double eventTime = frameStartTime + frac * dtFrameSeconds;
                    events[slot] = Event{static_cast<uint16_t>(x), static_cast<uint16_t>(y),
                                         eventTime, static_cast<int8_t>(on ? 1 : -1)};
                }

                startT = tEndIdeal;
                deltaVdLegacy = 0.0;
            }

            // If we fell out of the loop by exhausting the cap (rather than the
            // normal "doesn't fit"/defensive-guard exits above), this pixel is
            // producing more crossings per frame than any sane configuration
            // should -- deltaVdLegacy is left at 0 (from the last successful
            // reset), which just means it starts the next frame fresh rather than
            // carrying a precise fractional residual. Cheap to detect if you want
            // visibility into it: atomicAdd a diagnostic counter here and check it
            // host-side; omitted to keep the hot path lean.
            (void)exhaustedCap;

            state.baseIntensity = static_cast<float>(newVal);
            state.deltaVdRes = static_cast<float>(deltaVdLegacy);
            states[idx] = state;
            rngStates[idx] = rng;
        }

#include "GenerateEventsKernelDVSStochastic4.inc"

    } // anonymous namespace

    EventGeneratorCUDA::EventGeneratorCUDA(const Config &config, int maxEventsPerFrame)
        : config_(config), k_(dvs::sensor_k_preset(config.sensorType)), maxEvents_(maxEventsPerFrame)
    {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    EventGeneratorCUDA::~EventGeneratorCUDA()
    {
        freeDeviceBuffers();
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    // Cuda kernel device memory buffer handling helper functions

    void EventGeneratorCUDA::freeDeviceBuffers()
    {
        if (d_image_)
            cudaFree(d_image_);
        if (d_pixelStates_)
            cudaFree(d_pixelStates_);
        if (d_outputBuffer_)
            cudaFree(d_outputBuffer_);
        if (d_rngStates_)
            cudaFree(d_rngStates_);
        if (h_image_)
            cudaFreeHost(h_image_);
        if (h_outputBuffer_)
            cudaFreeHost(h_outputBuffer_);
        d_image_ = nullptr;
        d_pixelStates_ = nullptr;
        d_outputBuffer_ = nullptr;
        d_rngStates_ = nullptr;
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

        // Only pay for RNG state when actually running the stochastic path --
        // this buffer is not small (width*height*sizeof(curandState)).
        if (!config_.fastDeterministicMode)
        {
            CUDA_CHECK(cudaMalloc(&d_rngStates_, numPixels * sizeof(curandState)));
        }

        // Pinned (page-locked) host memory -- pageable memcpy latency is one
        // of the biggest hidden sources of frame-to-frame jitter; pinned
        // memory gets a direct DMA transfer with much tighter, repeatable
        // timing.
        CUDA_CHECK(cudaHostAlloc(&h_image_, numPixels * sizeof(uint16_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_outputBuffer_, outputBufferBytes_, cudaHostAllocDefault));
    }

    // EventGeneratorCUDA helper functions
    void EventGeneratorCUDA::initialize(int width, int height)
    {
        freeDeviceBuffers();
        width_ = width;
        height_ = height;
        allocateDeviceBuffers();
        initialized_ = false;
        // vector kernels need width % 4 == 0 (aligned 8B/16B loads); 1280 qualifies
        useVec4_ = (width_ % 4 == 0) && (std::getenv("EVSIM_NO_VEC4") == nullptr);
    }

    void EventGeneratorCUDA::reset()
    {
        initialized_ = false;
    }

    // EventGeneratorCuda main runner function

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

        // Mirrors EventGenerator::generate()'s lazy-init behavior: if nobody
        // called initialize() explicitly yet, do it now from the width/height
        // passed in here. CNvSIPLConsumer calls initialize() explicitly before
        // any generate() call, so this is a safety net, not the normal path -
        // note it does NOT handle a resolution change mid-stream (neither does
        // the CPU version): call initialize() yourself for that.
        if (width_ == 0 || height_ == 0)
            initialize(width, height);

        if (width_ == 0 || height_ == 0)
            return packet;

        // Stage into pinned memory row-by-row (handles stridePixels != width_).
        // If your upstream downscale step can write directly into a pinned
        // buffer, pass that pointer as `image` with stridePixels == width_ and
        // this memcpy collapses to a single cudaMemcpyAsync with no staging.
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
        const int cuda_blocks = 32;
        const int cuda_threads = 8;
        const dim3 block(cuda_blocks, cuda_threads);
        const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
        const dim3 grid4(((width_ / 4) + block.x - 1) / block.x,
                         (height_ + block.y - 1) / block.y);

        if (!initialized_)
        {
            InitFirstFrameKernel<<<grid, block, 0, stream_>>>(
                d_pixelStates_, d_image_, width_, height_, width_);
            if (!config_.fastDeterministicMode)
            {
                RNGInitKernel<<<grid, block, 0, stream_>>>(
                    d_rngStates_, width_, height_, static_cast<unsigned long long>(rngSeed_));
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream_));

            initialized_ = true;
            previousTimestamp_ = timestamp;
            packet.startTime = timestamp;
            packet.endTime = timestamp;
            packet.frameNumber = frameNumber_++;
            return packet;
        }

        // Guard against a non-increasing timestamp (jitter/repeats from real
        // hardware clocks) the same way the CPU EventGenerator does -- nudge it
        // forward slightly rather than dividing by zero/negative dt below.
        double dtFrameSeconds = timestamp - previousTimestamp_;
        if (dtFrameSeconds <= 0.0)
        {
            dtFrameSeconds = 1e-6;
        }
        // Sensor K coefficients (see dvs_types.hpp / sensor_k_preset) are
        // calibrated assuming timestamps in MICROSECONDS -- convert here so the
        // drift math matches the CPU reference regardless of this project's
        // native double-seconds Event::timestamp convention. Event timestamps
        // themselves are still produced in seconds below (frameStartTime +
        // frac * dtFrameSeconds), since `frac` is a dimensionless ratio.
        const double dtFrameUs = dtFrameSeconds * 1e6;

        // Reset only the leading event-count int, not the whole output buffer.
        CUDA_CHECK(cudaMemsetAsync(d_outputBuffer_, 0, sizeof(int), stream_));

        if (config_.fastDeterministicMode)
        {
            if (useVec4_)
                GenerateEventsKernelDVSFast4<<<grid, block, 0, stream_>>>(
                d_pixelStates_, d_image_, width_, height_, width_,
                static_cast<float>(k_.k1 / dtFrameUs),
                static_cast<float>(k_.k2), static_cast<float>(k_.k4), static_cast<float>(k_.k5),
                static_cast<float>(config_.contrastThresholdOn),
                static_cast<float>(config_.contrastThresholdOff),
                static_cast<float>(dtFrameUs), dtFrameSeconds, previousTimestamp_,
                maxEvents_, d_outputBuffer_);

            else
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
            if (useVec4_)
                GenerateEventsKernelDVSStochastic4<<<grid, block, 0, stream_>>>(
                d_pixelStates_, d_rngStates_, d_image_, width_, height_, width_,
                static_cast<float>(k_.k1), static_cast<float>(k_.k2), static_cast<float>(k_.k3),
                static_cast<float>(k_.k4), static_cast<float>(k_.k5), static_cast<float>(k_.k6),
                static_cast<float>(config_.contrastThresholdOn),
                static_cast<float>(config_.contrastThresholdOff),
                static_cast<float>(dtFrameUs), dtFrameSeconds, previousTimestamp_,
                maxEvents_, d_outputBuffer_);

            else
                GenerateEventsKernelDVSStochastic<<<grid, block, 0, stream_>>>(
                d_pixelStates_, d_rngStates_, d_image_, width_, height_, width_,
                static_cast<float>(k_.k1), static_cast<float>(k_.k2), static_cast<float>(k_.k3),
                static_cast<float>(k_.k4), static_cast<float>(k_.k5), static_cast<float>(k_.k6),
                static_cast<float>(config_.contrastThresholdOn),
                static_cast<float>(config_.contrastThresholdOff),
                static_cast<float>(dtFrameUs), dtFrameSeconds, previousTimestamp_,
                maxEvents_, d_outputBuffer_);


        }
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(prof_.ev[2], stream_); // [prof] after kernel

        // Fixed-size D2H copy every single frame (count + full event capacity)
        // -- this is the key change for stable timing. A copy sized to the
        // ACTUAL event count would make transfer time scene-dependent, which
        // is exactly the kind of variability you're trying to remove.
        /*
        CUDA_CHECK(cudaMemcpyAsync(h_outputBuffer_, d_outputBuffer_, outputBufferBytes_,
                                    cudaMemcpyDeviceToHost, stream_));
        cudaEventRecord(prof_.ev[3], stream_); // [prof] after D2H
        const double tY0 = nowMs();            // [prof]
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        prof_.hostSync = nowMs() - tY0;        // [prof]
        */

        // Step 1: just the 4-byte count
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
        // int eventCount = 0;
        // std::memcpy(&eventCount, h_outputBuffer_, sizeof(int));
        // if (eventCount > maxEvents_)
        // {
        //     fprintf(stderr, "EventGeneratorCUDA: frame produced %d events, capacity is %d "
        //                     "- %d events dropped. Raise maxEventsPerFrame.\n",
        //             eventCount, maxEvents_, eventCount - maxEvents_);
        //     eventCount = maxEvents_;
        // }

        const Event *devEvents = reinterpret_cast<const Event *>(h_outputBuffer_ + kHeaderBytes);

        packet.startTime = previousTimestamp_;
        packet.endTime = timestamp;
        packet.frameNumber = frameNumber_++;
        const double tA0 = nowMs();                              // [prof]
        packet.events.assign(devEvents, devEvents + eventCount); // O(eventCount), not O(W*H)
        prof_.hostAssign = nowMs() - tA0;                        // [prof]

        previousTimestamp_ = timestamp;

        prof_.hostTotal = nowMs() - tFn0; // [prof]
        prof_.endFrame(static_cast<size_t>(eventCount), d2hBytes, h2dBytes);
        return packet;
    }

} // namespace evsim
