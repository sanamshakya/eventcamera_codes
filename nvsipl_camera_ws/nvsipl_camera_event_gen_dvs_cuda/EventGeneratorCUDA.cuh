#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

// IMPORTANT: use the project's REAL headers here, not local stand-ins.
// Config.h must now define (see the evsim_integration Config.h delivered
// earlier in this project):
//   - std::string sensorType         (selects dvs::sensor_k_preset)
//   - bool        fastDeterministicMode
//   - double      contrastThresholdOn / contrastThresholdOff
// positiveThreshold/negativeThreshold (the old linear-model fields) are no
// longer read by this class -- they were the previous model's thresholds,
// not this one's.
#include "Config.h"
#include "Event.h"
#include "EventPacket.h"
#include "IEventGenerator.h"

// dvs_core: only dvs_types.hpp is needed here (SensorK + sensor_k_preset are
// header-only, host-side helpers) -- no link dependency on event_sim.cpp/.o.
// Make sure dvs_core's include/ dir is on this target's include path (both
// the nvcc compile of the .cu and any .cpp that includes this .cuh).
#include "dvs_types.hpp"

namespace evsim
{

// DVS-Voltmeter per-pixel state. Replaces the old {referenceIntensity,
// referenceTime} pair: this model compares against the PREVIOUS FRAME's raw
// intensity (updated unconditionally every call, not just on a crossing) and
// carries a residual sub-threshold voltage across frames -- see
// dvs::EventSim::generate_events() (event_sim.cpp) for the CPU/OpenMP
// reference this mirrors. Frame timing (previousTimestamp_) is tracked
// host-side instead of per-pixel -- see the note in the .cu.
struct PixelStateGPU
{
    float baseIntensity; // previous frame's raw pixel value (dvs_core: base_frame_)
    float deltaVdRes;    // residual sub-threshold voltage carried forward (dvs_core: delta_vd_res_)
};

class EventGeneratorCUDA : public IEventGenerator
{
public:
    // maxEventsPerFrame: hard cap on events/frame. Sized this way (not
    // grown dynamically) is what keeps the D2H copy a fixed size every
    // frame -- see generate(). Tune to your scene's worst-case motion.
    explicit EventGeneratorCUDA(const Config &config, int maxEventsPerFrame = 200000);
    ~EventGeneratorCUDA() override;

    EventGeneratorCUDA(const EventGeneratorCUDA &) = delete;
    EventGeneratorCUDA &operator=(const EventGeneratorCUDA &) = delete;

    void initialize(int width, int height) override;
    void reset() override;

    // image: HOST pointer. stridePixels: row stride of `image` in
    // uint16_t elements (matches your existing CaptureRawBayerBuffer /
    // downscale interface). Pass a pointer into PINNED memory here if you
    // can (see notes in the .cu) -- it removes another source of jitter.
    //
    // width/height here match IEventGenerator's signature so this is a
    // drop-in replacement for EventGenerator in CNvSIPLConsumer: they are
    // only used to lazily initialize() on the first call if you haven't
    // called initialize() explicitly already (CNvSIPLConsumer does call it
    // explicitly today, so in practice these are only a safety net).
    EventPacket generate(
        const uint16_t *image,
        int width,
        int height,
        int stridePixels,
        double timestamp) override;

private:
    void allocateDeviceBuffers();
    void freeDeviceBuffers();

    Config config_;
    dvs::SensorK k_; // resolved once from config_.sensorType in the constructor
    int width_ = 0;
    int height_ = 0;
    int maxEvents_ = 0;
    bool initialized_ = false;
    uint64_t frameNumber_ = 0;
    uint64_t rngSeed_ = 0xC0FFEEu; // only used in stochastic mode

    // Frame timing tracked host-side (this replaces the old per-pixel
    // referenceTime + the h_pixel0_ device->host snapshot copy entirely --
    // the previous frame's timestamp is already known to the host, no need
    // to round-trip it through the GPU). See generate() in the .cu.
    double previousTimestamp_ = 0.0;

    cudaStream_t stream_ = nullptr;

    // Device buffers (persistent across frames -- allocated once in
    // initialize(), never per-frame. Per-frame cudaMalloc/cudaFree is a
    // major, avoidable source of latency variance.)
    uint16_t       *d_image_       = nullptr; // width_*height_, tightly packed
    PixelStateGPU  *d_pixelStates_ = nullptr; // width_*height_
    uint8_t        *d_outputBuffer_ = nullptr; // [int eventCount][Event x maxEvents_]

    // Only allocated when !config_.fastDeterministicMode -- the deterministic
    // path never touches an RNG, so there's no reason to pay for this buffer
    // (width_*height_*sizeof(curandState), which is not small) when running fast.
    curandState *d_rngStates_ = nullptr;

    // Pinned host mirrors -- required for predictable-latency DMA.
    uint16_t *h_image_        = nullptr;
    uint8_t  *h_outputBuffer_ = nullptr;

    size_t outputBufferBytes_ = 0;
};

} // namespace evsim
