#pragma once

#include <cstdint>

#include "EventPacket.h"

namespace evsim
{

// Common interface implemented by both EventGenerator (CPU, row-threaded)
// and EventGeneratorCUDA (GPU). Lets CNvSIPLConsumer (or anything else
// driving one of these) hold a single pointer type and pick the backend
// once at construction time -- see
// CNvSIPLConsumer::EnableEventGeneration().
class IEventGenerator
{
public:
    virtual ~IEventGenerator() = default;

    virtual void initialize(int width, int height) = 0;
    virtual void reset() = 0;

    // image        pointer to width*height pixel values, row-major
    // stridePixels row stride in PIXELS (not bytes) - pass width if the
    //              buffer is tightly packed (no row padding)
    //
 
    virtual EventPacket generate(
        const uint16_t *image,
        int width,
        int height,
        int stridePixels,
        double timestamp) = 0;

    //------------------------------------------------------------------
    // Optional CLAHE pre-processing hooks.
    //
    // Default implementations are no-ops so the CPU backend (and anything
    // else implementing this interface) keeps compiling unchanged. Only
    // EventGeneratorCUDA overrides them. supportsClahe() lets the caller
    // detect the no-op case and warn instead of silently doing nothing.
  
    //------------------------------------------------------------------
    virtual bool supportsClahe() const { return false; }
    virtual void setClaheEnabled(bool /*on*/) {}
    virtual void setClaheClip(float /*clipLimit*/) {}

    // bits        : effective bit depth of the pixel values CLAHE equalizes.
    //               8  -> 256-bin histogram  (8-bit video sources)
    //               12 -> 4096-bin histogram (12-bit sensor data)
    // inputShift  : right-shift applied to each input pixel before binning.
    //               0 for right-justified data (12-bit value in 0..4095),
    //               4 for 12-bit data left-justified in 16 bits (max ~65520).
    // Output is always on the 0..(2^bits - 1) scale.
    virtual void setClaheFormat(int /*bits*/, int /*inputShift*/) {}
};

} // namespace evsim
