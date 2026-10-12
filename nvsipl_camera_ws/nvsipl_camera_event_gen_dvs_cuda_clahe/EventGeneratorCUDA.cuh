#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <atomic>
#include <memory>


#include "ClaheCuda.cuh"
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
// carries a residual sub-threshold voltage across frames
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

    //------------------------------------------------------------------
    // Optional on-GPU CLAHE pre-processing (off by default).
    //
  
    //------------------------------------------------------------------
    bool  supportsClahe() const override { return true; }
    void  setClaheEnabled(bool on) override;
    bool  claheEnabled() const { return claheEnabled_.load(); }
    void  setClaheClip(float clipLimit) override;
    float claheClip() const { return claheClip_.load(); }
    // bits: 8 or 12.  inputShift: 0 for right-justified, 4 for 12-bit data
    // left-justified in 16 bits.
    void  setClaheFormat(int bits, int inputShift) override;

    // If true, generate() also copies the frame the event kernels saw
    // (post-CLAHE) back to the host. Costs one extra W*H*2-byte D2H per frame,
    // so enable it only when you want to display/inspect that frame.
    void setCaptureInputFrame(bool on) { captureInput_ = on; }

    // Pinned host buffer, width x height, uint16 (values 0..255), tightly
    // packed. Valid until the next generate(); nullptr unless
    // setCaptureInputFrame(true) and a frame has been generated.
    const uint16_t *lastInputFrame() const { return inputValid_ ? h_inputCopy_ : nullptr; }

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
    uint16_t       *d_image_       = nullptr; // width_*height_, tightly packed (what the event kernels read)
    PixelStateGPU  *d_pixelStates_ = nullptr; // width_*height_
    uint8_t        *d_outputBuffer_ = nullptr; // [int eventCount][Event x maxEvents_]

    curandState *d_rngStates_ = nullptr;

    // Pinned host mirrors -- required for predictable-latency DMA.
    uint16_t *h_image_        = nullptr;
    uint8_t  *h_outputBuffer_ = nullptr;

    size_t outputBufferBytes_ = 0;

    // ---- CLAHE ----
    std::unique_ptr<ClaheCuda> clahe_;
    uint16_t *d_imageRaw_   = nullptr; // un-enhanced upload target when CLAHE is on
    uint16_t *h_inputCopy_  = nullptr; // pinned; optional copy of d_image_ for display
    // Written by any thread (setters), read by generate().
    std::atomic<bool>  claheEnabled_{false};
    std::atomic<float> claheClip_{2.0f};
    std::atomic<int>   claheBits_{12};   // default: NVSIPL RAW12 in 16-bit container
    std::atomic<int>   claheShift_{0};   // 0 = right-justified
    std::atomic<bool>  resetRequested_{false};
    bool  captureInput_   = false;
    bool  inputValid_     = false;
};

} // namespace evsim
