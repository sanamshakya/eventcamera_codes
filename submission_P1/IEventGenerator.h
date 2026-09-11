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
    // NOTE: both implementations only consume width/height to lazily call
    // initialize() if you haven't already called it explicitly. Passing a
    // different width/height on a later call does NOT trigger a
    // resolution change in either implementation -- call initialize()
    // explicitly for that (as CNvSIPLConsumer does today).
    virtual EventPacket generate(
        const uint16_t *image,
        int width,
        int height,
        int stridePixels,
        double timestamp) = 0;
};

} // namespace evsim
