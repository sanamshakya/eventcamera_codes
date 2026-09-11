#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "EventPacket.h"

// Deliberately independent, straightforward, single-threaded re-statement
// of the same per-pixel algorithm as the CUDA kernel. This is the "golden
// model" the test compares GPU output against -- keep this simple and
// obviously-correct rather than clever, since its whole job is to be a
// trustworthy oracle.

namespace evsim
{

struct Config; // fwd-decl matches EventGeneratorCUDA.cuh's Config

struct RefPixelState
{
    float  referenceIntensity = 0.0f;
    double referenceTime = 0.0;
};

class EventGeneratorRef
{
public:
    EventGeneratorRef(float positiveThreshold, float negativeThreshold)
        : positiveThreshold_(positiveThreshold), negativeThreshold_(negativeThreshold)
    {
    }

    void initialize(int width, int height)
    {
        width_ = width;
        height_ = height;
        states_.assign(static_cast<size_t>(width_) * height_, RefPixelState{});
        initialized_ = false;
        frameNumber_ = 0;
    }

    // image: host pointer, stridePixels: row stride in uint16_t elements.
    // Same signature as EventGeneratorCUDA::generate() so both can be
    // driven by identical test-harness code.
    EventPacket generate(const uint16_t *image, int stridePixels, double timestamp)
    {
        EventPacket packet{};
        if (image == nullptr || width_ == 0 || height_ == 0)
            return packet;

        if (!initialized_)
        {
            for (int y = 0; y < height_; ++y)
            {
                const uint16_t *row = image + static_cast<size_t>(y) * stridePixels;
                for (int x = 0; x < width_; ++x)
                {
                    states_[y * width_ + x] =
                        RefPixelState{ static_cast<float>(row[x]), timestamp };
                }
            }
            initialized_ = true;
            packet.startTime = timestamp;
            packet.endTime = timestamp;
            packet.frameNumber = frameNumber_++;
            return packet;
        }

        packet.startTime = states_[0].referenceTime;
        packet.endTime = timestamp;
        packet.frameNumber = frameNumber_++;

        for (int y = 0; y < height_; ++y)
        {
            const uint16_t *row = image + static_cast<size_t>(y) * stridePixels;
            for (int x = 0; x < width_; ++x)
            {
                RefPixelState &state = states_[y * width_ + x];
                const float currentIntensity = static_cast<float>(row[x]);
                const float delta = currentIntensity - state.referenceIntensity;

                if (delta >= positiveThreshold_)
                {
                    const int crossings = static_cast<int>(std::floor(delta / positiveThreshold_));
                    if (crossings > 0)
                    {
                        const float invDelta = 1.0f / delta;
                        for (int i = 0; i < crossings; ++i)
                        {
                            const float crossingValue =
                                state.referenceIntensity + (i + 1) * positiveThreshold_;
                            float alpha = (crossingValue - state.referenceIntensity) * invDelta;
                            alpha = std::min(std::max(alpha, 0.0f), 1.0f);
                            const double eventTime =
                                state.referenceTime + alpha * (timestamp - state.referenceTime);
                            packet.events.push_back(
                                Event{ static_cast<uint16_t>(x), static_cast<uint16_t>(y),
                                       eventTime, +1 });
                        }
                        state.referenceIntensity += crossings * positiveThreshold_;
                        state.referenceTime = timestamp;
                    }
                }
                else if (delta <= negativeThreshold_)
                {
                    const int crossings =
                        static_cast<int>(std::floor((-delta) / (-negativeThreshold_)));
                    if (crossings > 0)
                    {
                        const float invNegDelta = 1.0f / (-delta);
                        for (int i = 0; i < crossings; ++i)
                        {
                            const float crossingValue =
                                state.referenceIntensity - (i + 1) * (-negativeThreshold_);
                            float alpha = (state.referenceIntensity - crossingValue) * invNegDelta;
                            alpha = std::min(std::max(alpha, 0.0f), 1.0f);
                            const double eventTime =
                                state.referenceTime + alpha * (timestamp - state.referenceTime);
                            packet.events.push_back(
                                Event{ static_cast<uint16_t>(x), static_cast<uint16_t>(y),
                                       eventTime, -1 });
                        }
                        state.referenceIntensity -= crossings * (-negativeThreshold_);
                        state.referenceTime = timestamp;
                    }
                }
            }
        }
        return packet;
    }

private:
    int width_ = 0, height_ = 0;
    bool initialized_ = false;
    uint64_t frameNumber_ = 0;
    float positiveThreshold_;
    float negativeThreshold_;
    std::vector<RefPixelState> states_;
};

} // namespace evsim
