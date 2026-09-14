#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include "Config.h"
#include "Event.h"
#include "EventPacket.h"
#include "IEventGenerator.h"

#include "dvs_types.hpp"

namespace evsim
{


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
