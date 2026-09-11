#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// NOTE: Align these with your real EventGenerator.h / EventPacket.h.
// Shown here only so this file is self-contained / compilable standalone.
#include "EventPacket.h"   // must define: struct Event{uint16_t x,y; double timestamp; int8_t polarity;};
                            //              struct EventPacket{double startTime,endTime; uint64_t frameNumber; std::vector<Event> events;};

namespace evsim
{

struct Config
{
    float positiveThreshold;
    float negativeThreshold;
    // bool useLogIntensity; // add to kernel's intensity() calc if you need this
};

struct PixelStateGPU
{
    float  referenceIntensity;
    double referenceTime;
};

class EventGeneratorCUDA
{
public:
    // maxEventsPerFrame: hard cap on events/frame. Sized this way (not
    // grown dynamically) is what keeps the D2H copy a fixed size every
    // frame -- see generate(). Tune to your scene's worst-case motion.
    explicit EventGeneratorCUDA(const Config &config, int maxEventsPerFrame = 200000);
    ~EventGeneratorCUDA();

    EventGeneratorCUDA(const EventGeneratorCUDA &) = delete;
    EventGeneratorCUDA &operator=(const EventGeneratorCUDA &) = delete;

    void initialize(int width, int height);
    void reset();

    // image: HOST pointer. stridePixels: row stride of `image` in
    // uint16_t elements (matches your existing CaptureRawBayerBuffer /
    // downscale interface). Pass a pointer into PINNED memory here if you
    // can (see notes in the .cu) -- it removes another source of jitter.
    EventPacket generate(const uint16_t *image, int stridePixels, double timestamp);

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
