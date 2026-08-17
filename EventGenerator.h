#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

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
     */
    EventPacket generate(
        const cv::Mat& gray,
        double timestamp);
        
    EventPacket generate(
    const uint16_t *image,
    int width,
    int height,
    int stridePixels,
    double timestamp);

private:

    inline float intensity(uint8_t value) const;

    void initializeFirstFrame(
        const cv::Mat& gray,
        double timestamp);

private:

    Config config_;

    int width_ = 0;

    int height_ = 0;

    bool initialized_ = false;

    std::vector<PixelState> pixels_;

    uint64_t frameNumber_ = 0;

};

} // namespace evsim
