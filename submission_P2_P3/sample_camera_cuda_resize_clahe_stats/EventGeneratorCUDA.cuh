#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

// IMPORTANT: use the project's REAL headers here, not local stand-ins.
// Config.h must define:
//   - std::string sensorType         (selects dvs::sensor_k_preset)
//   - bool        fastDeterministicMode
//   - double      contrastThresholdOn / contrastThresholdOff
#include "ClaheCuda.cuh"
#include "Config.h"
#include "Event.h"
#include "EventPacket.h"
#include "IEventGenerator.h"
#include "GenProfiler.h"

// dvs_core: only dvs_types.hpp is needed here (SensorK + sensor_k_preset are
// header-only, host-side helpers).
#include "dvs_types.hpp"

namespace evsim
{

// DVS-Voltmeter per-pixel state (see dvs::EventSim::generate_events()).
// alignas(8) lets the compiler use one 64-bit load/store per pixel state.
struct alignas(8) PixelStateGPU
{
    float baseIntensity; // previous frame's raw pixel value (dvs_core: base_frame_)
    float deltaVdRes;    // residual sub-threshold voltage carried forward (dvs_core: delta_vd_res_)
};

// ---------------------------------------------------------------------------
// Stochastic-path instrumentation
// ---------------------------------------------------------------------------

// Safety cap on threshold crossings simulated per pixel per frame.
constexpr int kStochMaxCrossings = 16;

enum StochPhase : int
{
    PH_G_LOAD = 0, PH_G_DRIFT, PH_G_TEST, PH_G_APPEND, PH_G_ADVANCE, // gate kernel
    PH_B_LOAD, PH_B_PROB, PH_B_SAMPLE, PH_B_EMIT, PH_B_STORE,        // event kernel
    PH_COUNT
};

// Zeroed once per frame (cudaMemsetAsync), copied back once per frame.
// numCandidates / numCapHit / numBadDeltaT are always maintained;
// iterHist / cycles are only filled when built with -DEVSIM_KERNEL_PROF.
struct StochCounters
{
    unsigned int numCandidates;
    unsigned int numCapHit;                        // pixels that hit kStochMaxCrossings
    unsigned int numBadDeltaT;                     // NaN / <=0 waiting times
    unsigned int iterHist[kStochMaxCrossings + 1]; // [k] = candidate pixels with exactly k events
    unsigned long long cycles[PH_COUNT];           // summed per-thread clock64 cycles
};

// Host-side accumulator, reported every 100 frames.
struct StochProfHost
{
    int frames = 0;
    double msGate = 0, msEvents = 0;
    unsigned long long cand = 0, capHit = 0, bad = 0;
    unsigned long long hist[kStochMaxCrossings + 1] = {};
    unsigned long long cyc[PH_COUNT] = {};
};

class EventGeneratorCUDA : public IEventGenerator
{
public:
    // maxEventsPerFrame: hard cap on events/frame (buffer is sized once, so
    // the output buffer never reallocates).
    explicit EventGeneratorCUDA(const Config &config, int maxEventsPerFrame = 200000);
    ~EventGeneratorCUDA() override;

    EventGeneratorCUDA(const EventGeneratorCUDA &) = delete;
    EventGeneratorCUDA &operator=(const EventGeneratorCUDA &) = delete;

    void initialize(int width, int height) override;
    void reset() override;

    // Stochastic path only: enable/disable the candidate gate and set its
    // confidence z (z=5 -> ~6e-7 missed-crossing probability per barrier per
    // pixel per frame). Disabled => every pixel is a candidate (exact path).
    void setGate(bool enabled, float z = 5.0f)
    {
        gateEnabled_ = enabled;
        gateZ_ = z;
    }

    // image: HOST pointer. stridePixels: row stride of `image` in uint16_t
    // elements. width/height only matter for lazy initialize().
    EventPacket generate(
        const uint16_t *image,
        int width,
        int height,
        int stridePixels,
        double timestamp) override;

    // dImage: DEVICE pointer to a pitch-linear uint16 image, pitchBytes in BYTES.
    // Copies GPU->GPU into the packed internal buffer and right-shifts by rawShift
    // (before CLAHE, if enabled). width/height must match initialize().
    EventPacket generateFromDevice(const uint16_t *dImage, int width, int height,
                                   size_t pitchBytes, double timestamp, int rawShift);

    //------------------------------------------------------------------
    // Optional on-GPU CLAHE pre-processing (off by default).
    //
    // Runs on the generator stream, right before the event kernels, on
    // the frame handed to generate()/generateFromDevice(). The setters
    // are thread-safe (atomics); a change restarts the per-pixel state on
    // the next frame, so the brightness jump between raw and enhanced
    // frames is not turned into a burst of events.
    //
    // These are not part of IEventGenerator: call them on the concrete
    // EventGeneratorCUDA.
    //------------------------------------------------------------------
    void  setClaheEnabled(bool on);
    bool  claheEnabled() const { return claheEnabled_.load(); }
    void  setClaheClip(float clipLimit);
    float claheClip() const { return claheClip_.load(); }
    // bits: 8 or 12 (histogram depth). inputShift: 0 when the data is already
    // on the 0..(2^bits-1) scale -- generateFromDevice() applies rawShift
    // BEFORE CLAHE, so pass 0 there. 4 would be for 12-bit data left-justified
    // in 16 bits fed straight to CLAHE.
    void  setClaheFormat(int bits, int inputShift);

private:
    void allocateDeviceBuffers();
    void freeDeviceBuffers();
    void accumulateStochProfile(size_t numPixels);
    // Enqueue CLAHE on stream_: d_imageRaw_ -> d_image_. No-op if !useClahe.
    void runClahe(bool useClahe, float clip, int bits, int shift);

    Config config_;
    dvs::SensorK k_; // resolved once from config_.sensorType in the constructor
    int width_ = 0;
    int height_ = 0;
    int maxEvents_ = 0;
    bool initialized_ = false;
    uint64_t frameNumber_ = 0;
    uint64_t rngSeed_ = 0xC0FFEEu; // stochastic mode: Philox seed

    // Frame timing tracked host-side.
    double previousTimestamp_ = 0.0;

    cudaStream_t stream_ = nullptr;

    // Device buffers (persistent across frames).
    uint16_t      *d_image_        = nullptr; // width_*height_, tightly packed
    PixelStateGPU *d_pixelStates_  = nullptr; // width_*height_
    uint8_t       *d_outputBuffer_ = nullptr; // [int eventCount][Event x maxEvents_]

    // Stochastic path only (the Philox RNG is stateless: no per-pixel RNG buffer).
    int           *d_candList_ = nullptr;     // width_*height_ pixel indices
    StochCounters *d_counters_ = nullptr;
    StochCounters *h_counters_ = nullptr;     // pinned

    // Pinned host mirrors.
    uint16_t *h_image_        = nullptr;
    uint8_t  *h_outputBuffer_ = nullptr;

    size_t outputBufferBytes_ = 0;

    // Stochastic path configuration / profiling.
    float gateZ_ = 5.0f;
    bool  gateEnabled_ = true;
    int   numSMs_ = 0;
    int floorMode_ = -1;
    cudaEvent_t sEv_[3] = {nullptr, nullptr, nullptr};
    StochProfHost sprof_;

    // ---- CLAHE ----
    std::unique_ptr<ClaheCuda> clahe_;
    uint16_t *d_imageRaw_ = nullptr; // un-enhanced frame when CLAHE is on; CLAHE writes d_image_
    // Written by any thread (setters), read once per frame by generate()/generateFromDevice().
    std::atomic<bool>  claheEnabled_{false};
    std::atomic<float> claheClip_{2.0f};
    std::atomic<int>   claheBits_{12}; // 12 covers 0..4095; use 8 when the input is shifted to 0..255
    std::atomic<int>   claheShift_{0};
    std::atomic<bool>  resetRequested_{false};

    // Shared simulation tail: d_image_ already holds the current (packed, and
    // enhanced when CLAHE is on) frame.
    EventPacket runFrame(double timestamp, double tFn0);

    GenProfiler prof_{100, 5}; // report every 100 frames, ignore first 5
};

} // namespace evsim
