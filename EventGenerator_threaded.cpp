#include "EventGenerator.h"
#include "EventPacket.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>

namespace evsim
{
    // Set to 1 to restore the old per-frame stdout logging. Leave 0 in the
    // real-time capture path -- iostream formatting/locale + endl flushes
    // are not free, and they were running on the SIPL callback thread.
#ifndef EVSIM_VERBOSE_LOG
#define EVSIM_VERBOSE_LOG 0
#endif

    EventGenerator::EventGenerator(const Config &config)
        : config_(config)
    {
        // Reserve worker threads once. hardware_concurrency() can return 0
        // on some platforms (spec allows it) so guard against that.
        unsigned int hwThreads = std::thread::hardware_concurrency();
        numThreads_ = hwThreads == 0 ? 4u : hwThreads;
    }

    void EventGenerator::initialize(
        int width,
        int height)
    {
        width_ = width;
        height_ = height;

        pixels_.clear();
        pixels_.resize(width_ * height_);

        // Cap threads to something sane relative to image size so we don't
        // spin up more threads than rows on small/tiled inputs.
        numThreads_ = std::min<unsigned int>(numThreads_,
                                              static_cast<unsigned int>(std::max(1, height_)));

        threadLocalEvents_.resize(numThreads_);

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
        return static_cast<float>(value);
    }

    void EventGenerator::initializeFirstFrame(
        const uint16_t *image,
        int stridePixels,
        double timestamp)
    {
#if EVSIM_VERBOSE_LOG
        std::cout << "Initializing first frame\n";
        std::cout << "Image size : " << width_ << " x " << height_ << std::endl;
#endif

        for (int y = 0; y < height_; ++y)
        {
            const uint16_t *row = image + static_cast<size_t>(y) * stridePixels;

            for (int x = 0; x < width_; ++x)
            {
                PixelState &pixel = pixels_[y * width_ + x];

                pixel.referenceIntensity = intensity(row[x]);
                pixel.referenceTime = timestamp;
                pixel.initialized = true;
            }
        }

        initialized_ = true;
    }

    // Processes rows [yStart, yEnd) for one worker thread. Each thread only
    // ever touches its own row range of pixels_, so there's no shared-state
    // race -- no locking needed here.
    void EventGenerator::processRowRange(
        const uint16_t *image,
        int stridePixels,
        double timestamp,
        int yStart,
        int yEnd,
        float positiveThreshold,
        float negativeThreshold,
        std::vector<Event> &outEvents)
    {
        outEvents.clear();
        outEvents.reserve(static_cast<size_t>(yEnd - yStart) * width_ / 10 + 16);

        // Precompute reciprocals once -- turns a division into a
        // multiplication inside the crossings loop.
        const float invPositive = 1.0f / positiveThreshold;
        const float invNegativeMag = 1.0f / (-negativeThreshold);

        for (int y = yStart; y < yEnd; ++y)
        {
            const uint16_t *row = image + static_cast<size_t>(y) * stridePixels;
            const int rowBase = y * width_;

            for (int x = 0; x < width_; ++x)
            {
                PixelState &pixel = pixels_[rowBase + x];

                const float currentIntensity = intensity(row[x]);
                const float delta = currentIntensity - pixel.referenceIntensity;

                if (delta >= positiveThreshold)
                {
                    const int crossings = static_cast<int>(std::floor(delta * invPositive));
                    const float invDelta = 1.0f / delta;

                    for (int i = 0; i < crossings; ++i)
                    {
                        const float crossingValue =
                            pixel.referenceIntensity + (i + 1) * positiveThreshold;

                        float alpha = (crossingValue - pixel.referenceIntensity) * invDelta;
                        alpha = std::clamp(alpha, 0.0f, 1.0f);

                        const double eventTime =
                            pixel.referenceTime + alpha * (timestamp - pixel.referenceTime);

                        outEvents.push_back(
                            {static_cast<uint16_t>(x),
                             static_cast<uint16_t>(y),
                             eventTime,
                             +1});
                    }

                    pixel.referenceIntensity += crossings * positiveThreshold;
                    pixel.referenceTime = timestamp;
                }
                else if (delta <= negativeThreshold)
                {
                    const int crossings =
                        static_cast<int>(std::floor((-delta) * invNegativeMag));
                    const float invNegDelta = 1.0f / (-delta);

                    for (int i = 0; i < crossings; ++i)
                    {
                        const float crossingValue =
                            pixel.referenceIntensity - (i + 1) * (-negativeThreshold);

                        float alpha = (pixel.referenceIntensity - crossingValue) * invNegDelta;
                        alpha = std::clamp(alpha, 0.0f, 1.0f);

                        const double eventTime =
                            pixel.referenceTime + alpha * (timestamp - pixel.referenceTime);

                        outEvents.push_back(
                            {static_cast<uint16_t>(x),
                             static_cast<uint16_t>(y),
                             eventTime,
                             -1});
                    }

                    pixel.referenceIntensity -= crossings * (-negativeThreshold);
                    pixel.referenceTime = timestamp;
                }
            }
        }
    }

    EventPacket
    EventGenerator::generate(
        const uint16_t *image,
        int width,
        int height,
        int stridePixels,
        double timestamp)
    {
#if EVSIM_VERBOSE_LOG
        static uint64_t frameCount = 0;
        std::cout << "\n========== Frame " << frameCount++ << " ==========\n";
        std::cout << "Timestamp : " << timestamp << std::endl;
#endif

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

        const float positiveThreshold = config_.positiveThreshold;
        const float negativeThreshold = config_.negativeThreshold;

        auto t0 = std::chrono::high_resolution_clock::now();

        //------------------------------------------
        // Partition rows across worker threads. Each thread owns a
        // disjoint row range of pixels_ and writes into its own
        // threadLocalEvents_[t] buffer -- no shared mutable state, so no
        // mutex/atomics are needed on the hot path.
        //------------------------------------------

        const unsigned int numThreads =
            std::min(numThreads_, static_cast<unsigned int>(std::max(1, height_)));

        const int rowsPerThread = (height_ + static_cast<int>(numThreads) - 1) /
                                   static_cast<int>(numThreads);

        std::vector<std::thread> workers;
        workers.reserve(numThreads);

        for (unsigned int t = 0; t < numThreads; ++t)
        {
            const int yStart = static_cast<int>(t) * rowsPerThread;
            const int yEnd = std::min(height_, yStart + rowsPerThread);

            if (yStart >= yEnd)
                continue;

            workers.emplace_back(
                &EventGenerator::processRowRange, this,
                image, stridePixels, timestamp,
                yStart, yEnd,
                positiveThreshold, negativeThreshold,
                std::ref(threadLocalEvents_[t]));
        }

        for (auto &w : workers)
            w.join();

        //------------------------------------------
        // Merge. Rows are contiguous per thread and threads are launched
        // in increasing row order, so concatenating in thread order keeps
        // events in the same raster-scan order the single-threaded
        // version produced (NOT strict time order in either version).
        //------------------------------------------

        size_t totalEvents = 0;
        for (const auto &v : threadLocalEvents_)
            totalEvents += v.size();

        packet.events.clear();
        packet.events.reserve(totalEvents);

        for (auto &v : threadLocalEvents_)
        {
            packet.events.insert(packet.events.end(),
                                  std::make_move_iterator(v.begin()),
                                  std::make_move_iterator(v.end()));
        }

        auto t1 = std::chrono::high_resolution_clock::now();

#if EVSIM_VERBOSE_LOG
        std::cout << "[Event generation time] : "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << "ms\n";

        std::cout << "\nPacket info\n";
        std::cout << "Start = " << packet.startTime << std::endl;
        std::cout << "End = " << packet.endTime << std::endl;

        if (!packet.events.empty())
        {
            std::cout << "First event = " << packet.events.front().timestamp << std::endl;
            std::cout << "Last event = " << packet.events.back().timestamp << std::endl;
        }
#else
        (void)t1;
#endif

        return packet;
    }

} // namespace evsim
