#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// IMPORTANT: use the project's REAL headers here, not local stand-ins.
// EventGenerator_threaded2.h already does `#include "Config.h"` +
// `#include "Event.h"` + `#include "EventPacket.h"` for the exact same
// types - if this file re-declared its own `struct evsim::Config` (as the
// original standalone version did), you'd get an ODR violation / duplicate
// symbol the moment both headers land in the same translation unit (which
// happens as soon as CNvSIPLConsumer includes both). Config.h must define
// at least `positiveThreshold` / `negativeThreshold` as float - verify the
// field names match what's used below if your real Config.h differs.
#include "Config.h"
#include "Event.h"
#include "EventPacket.h"
#include "IEventGenerator.h"

namespace evsim
{

struct PixelStateGPU
{
    float  referenceIntensity;
    double referenceTime;
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
    int width_ = 0;
    int height_ = 0;
    int maxEvents_ = 0;
    bool initialized_ = false;
    uint64_t frameNumber_ = 0;

    cudaStream_t stream_ = nullptr;

    // Device buffers (persistent across frames -- allocated once in
    // initialize(), never per-frame. Per-frame cudaMalloc/cudaFree is a
    // major, avoidable source of latency variance.)
    uint16_t       *d_image_       = nullptr; // width_*height_, tightly packed
    PixelStateGPU  *d_pixelStates_ = nullptr; // width_*height_
    uint8_t        *d_outputBuffer_ = nullptr; // [int eventCount][Event x maxEvents_]

    // Pinned host mirrors -- required for predictable-latency DMA.
    uint16_t *h_image_        = nullptr;
    uint8_t  *h_outputBuffer_ = nullptr;
    PixelStateGPU *h_pixel0_  = nullptr; // pre-update snapshot of pixelStates[0], for packet.startTime

    size_t outputBufferBytes_ = 0;
};

} // namespace evsim
