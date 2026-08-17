#pragma once

#include <vector>
#include <cstdint>

#include "Config.h"
#include "Event.h"
#include "PixelState.h"
#include "EventPacket.h"

namespace evsim
{

class EventGenerator
{
public:

    explicit EventGenerator(const Config& config);

    /**
     * Initialize internal state.
     */
    void initialize(int width, int height);

    /**
     * Reset simulator state.
     */
    void reset();

    /**
     * Generate events for one frame.
     *
     * image        pointer to width*height pixel values, row-major
     * stridePixels row stride in PIXELS (not bytes) - pass width if the
     *              buffer is tightly packed (no row padding)
     */
    EventPacket generate(
        const uint16_t *image,
        int width,
        int height,
        int stridePixels,
        double timestamp);
        
     Config config_;

private:

    inline float intensity(uint16_t value) const;

    // Matches generate()'s pointer+stride contract - no data copy.
    void initializeFirstFrame(
        const uint16_t *image,
        int stridePixels,
        double timestamp);

private:

    

    int width_ = 0;

    int height_ = 0;

    bool initialized_ = false;

    std::vector<PixelState> pixels_;

    uint64_t frameNumber_ = 0;

};

} // namespace evsim
