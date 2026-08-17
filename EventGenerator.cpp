#include "EventGenerator.h"
#include "EventPacket.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace evsim
{

    EventGenerator::EventGenerator(const Config &config)
        : config_(config)
    {
    }

    void EventGenerator::initialize(
        int width,
        int height)
    {
        width_ = width;
        height_ = height;

        pixels_.clear();
        pixels_.resize(width_ * height_);

        initialized_ = false;
    }

    void EventGenerator::reset()
    {
        initialized_ = false;

        pixels_.clear();

        pixels_.resize(width_ * height_);
    }

    float EventGenerator::intensity(
        uint16_t value) const
    {
        // if(config_.useLogIntensity)
        // {
        //     return std::log(static_cast<float>(value) + 1.0f);
        // }

        return static_cast<float>(value);
    }

    void EventGenerator::initializeFirstFrame(
        const uint16_t *image,
        int stridePixels,
        double timestamp)
    {
        std::cout << "Initializing first frame\n";
        std::cout << "Image size : "
                  << width_
                  << " x "
                  << height_
                  << std::endl;

        for (int y = 0; y < height_; ++y)
        {
            const uint16_t *row = image + static_cast<size_t>(y) * stridePixels;

            for (int x = 0; x < width_; ++x)
            {
                PixelState &pixel =
                    pixels_[y * width_ + x];

                pixel.referenceIntensity =
                    intensity(row[x]);

                pixel.referenceTime =
                    timestamp;

                pixel.initialized = true;
            }
        }

        initialized_ = true;
    }

    EventPacket
    EventGenerator::generate(
        const uint16_t *image,
        int width,
        int height,
        int stridePixels,
        double timestamp)
    {
        static uint64_t frameCount = 0;

        std::cout << "\n========== Frame "
                  << frameCount++
                  << " ==========\n";

        std::cout << "Timestamp : "
                  << timestamp
                  << std::endl;

        EventPacket packet;

        if (image == nullptr)
            return packet;

        if (width_ == 0 || height_ == 0)
            initialize(width, height);

        //------------------------------------------
        // First frame only initializes references
        //------------------------------------------

        if (!initialized_)
        {
            initializeFirstFrame(image, stridePixels, timestamp);

            packet.startTime = timestamp;
            packet.endTime = timestamp;
            packet.frameNumber = frameNumber_++;

            return packet;
        }

        //------------------------------------------
        // Packet metadata
        //------------------------------------------

        packet.startTime = pixels_[0].referenceTime;
        packet.endTime = timestamp;
        packet.frameNumber = frameNumber_++;

        packet.events.reserve(width_ * height_ / 10);

        const float positiveThreshold = config_.positiveThreshold;
        const float negativeThreshold = config_.negativeThreshold;

        for (int y = 0; y < height_; ++y)
        {
            const uint16_t *row = image + static_cast<size_t>(y) * stridePixels;

            for (int x = 0; x < width_; ++x)
            {
                PixelState &pixel = pixels_[y * width_ + x];

                float currentIntensity = intensity(row[x]);

                float delta =
                    currentIntensity -
                    pixel.referenceIntensity;

                //=================================================
                // DEBUG: Print one pixel only
                //=================================================

                // const int debugX = width_ / 2;
                // const int debugY = height_ / 2;

                // if (x == debugX && y == debugY)
                // {
                //     std::cout
                //         << "Pixel(" << x << "," << y << ") "
                //         << "Raw=" << static_cast<int>(row[x])
                //         << " Current=" << currentIntensity
                //         << " Reference=" << pixel.referenceIntensity
                //         << " Delta=" << delta
                //         << std::endl;
                // }

                // std::cout
                //     << "Delta = " << delta
                //     << "  Positive Threshold = " << positiveThreshold
                //     << std::endl;

                //--------------------------------------------------
                // Positive events
                //--------------------------------------------------

                if (delta >= positiveThreshold)
                {

                    // std::cout << "Entered positive branch\n";

                    int crossings =
                        static_cast<int>(
                            std::floor(delta / positiveThreshold));

                    // std::cout
                    //     << "Crossings = "
                    //     << crossings
                    //     << std::endl;

                    for (int i = 0; i < crossings; ++i)
                    {
                        float crossingValue =
                            pixel.referenceIntensity +
                            (i + 1) * positiveThreshold;

                        float alpha =
                            (crossingValue -
                             pixel.referenceIntensity) /
                            delta;

                        alpha = std::clamp(alpha, 0.0f, 1.0f);

                        double eventTime =
                            pixel.referenceTime +
                            alpha *
                                (timestamp - pixel.referenceTime);

                        // std::cout
                        //     << "ON Event "
                        //     << "("
                        //     << x
                        //     << ","
                        //     << y
                        //     << ") "
                        //     << eventTime
                        //     << std::endl;

                        packet.events.push_back(
                            {static_cast<uint16_t>(x),
                             static_cast<uint16_t>(y),
                             eventTime,
                             +1});
                    }

                    pixel.referenceIntensity +=
                        crossings * positiveThreshold;

                    pixel.referenceTime = timestamp;
                }

                //--------------------------------------------------
                // Negative events
                //--------------------------------------------------

                else if (delta <= negativeThreshold)
                {
                    int crossings =
                        static_cast<int>(
                            std::floor((-delta) /
                                       (-negativeThreshold)));

                    for (int i = 0; i < crossings; ++i)
                    {
                        float crossingValue =
                            pixel.referenceIntensity -
                            (i + 1) * (-negativeThreshold);

                        float alpha =
                            (pixel.referenceIntensity -
                             crossingValue) /
                            (-delta);

                        alpha = std::clamp(alpha, 0.0f, 1.0f);

                        double eventTime =
                            pixel.referenceTime +
                            alpha *
                                (timestamp - pixel.referenceTime);

                        // std::cout
                        //     << "OFF Event "
                        //     << "("
                        //     << x
                        //     << ","
                        //     << y
                        //     << ") "
                        //     << eventTime
                        //     << std::endl;

                        packet.events.push_back(
                            {static_cast<uint16_t>(x),
                             static_cast<uint16_t>(y),
                             eventTime,
                             -1});
                    }

                    pixel.referenceIntensity -=
                        crossings * (-negativeThreshold);

                    pixel.referenceTime = timestamp;
                }
            }
        }

        // std::cout
        //     << "Generated Events = "
        //     << packet.events.size()
        //     << std::endl;

        std::cout << "\nPacket info\n";

        std::cout << "Start = "
                  << packet.startTime
                  << std::endl;

        std::cout << "End = "
                  << packet.endTime
                  << std::endl;

        if (!packet.events.empty())
        {
            std::cout
                << "First event = "
                << packet.events.front().timestamp
                << std::endl;

            std::cout
                << "Last event = "
                << packet.events.back().timestamp
                << std::endl;
        }

        return packet;
    }

} // namespace evsim
