#include "EventGenerator.h"
#include "EventPacket.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <chrono>

    // Set to 1 to restore the old per-frame stdout logging. Leave 0 in the
    // real-time capture path -- iostream formatting/locale + endl flushes
    // are not free, and they were running on the SIPL callback thread.
#ifndef EVSIM_VERBOSE_LOG
#define EVSIM_VERBOSE_LOG 0
#endif

namespace evsim
{
    EventGenerator::EventGenerator(const Config &config)
        : config_(config)
    {
        unsigned int hwThreads = std::thread::hardware_concurrency();
        numThreads_ = hwThreads == 0 ? 4u : hwThreads;
        // Pool is NOT started here -- it needs height_ (to size row ranges
        // and cap thread count sanely), which isn't known until the first
        // initialize() call.
    }

    EventGenerator::~EventGenerator()
    {
        stopPool();
    }

    void EventGenerator::stopPool()
    {
        if (pool_.empty())
            return;

        {
            std::lock_guard<std::mutex> lock(mtx_);
            stop_ = true;
            ++generation_;
        }
        cvStart_.notify_all();

        for (auto &t : pool_)
            if (t.joinable())
                t.join();

        pool_.clear();
        stop_ = false;
    }

    void EventGenerator::startPool()
    {
        // Called only from initialize(), after width_/height_/numThreads_
        // are finalized and pixels_ is sized. Any previous pool (e.g. from
        // a resolution change) has already been stopped by initialize().
        rowRanges_.resize(numThreads_);

        const int rowsPerThread = (height_ + static_cast<int>(numThreads_) - 1) /
                                   static_cast<int>(numThreads_);

        for (unsigned int t = 0; t < numThreads_; ++t)
        {
            const int yStart = static_cast<int>(t) * rowsPerThread;
            const int yEnd = std::min(height_, yStart + rowsPerThread);
            rowRanges_[t] = {yStart, yEnd};
        }

        lastSeenGeneration_.assign(numThreads_, 0);

        pool_.reserve(numThreads_);
        for (unsigned int t = 0; t < numThreads_; ++t)
            pool_.emplace_back(&EventGenerator::workerLoop, this, t);
    }

    void EventGenerator::workerLoop(unsigned int idx)
    {
        uint64_t lastGen = 0;

        while (true)
        {
            const uint16_t *image;
            int stridePixels;
            double timestamp;
            float positiveThreshold;
            float negativeThreshold;
            int yStart, yEnd;

            {
                std::unique_lock<std::mutex> lock(mtx_);
                cvStart_.wait(lock, [&]
                              { return stop_ || generation_ != lastGen; });

                if (stop_)
                    return;

                lastGen = generation_;

                // Snapshot the shared per-frame inputs while holding the
                // lock. The main thread only mutates these before bumping
                // generation_ and notifying, so this happens-before is safe.
                image = image_;
                stridePixels = stridePixels_;
                timestamp = timestamp_;
                positiveThreshold = positiveThreshold_;
                negativeThreshold = negativeThreshold_;
                yStart = rowRanges_[idx].first;
                yEnd = rowRanges_[idx].second;
            }

            processRowRange(image, stridePixels, timestamp,
                             yStart, yEnd,
                             positiveThreshold, negativeThreshold,
                             threadLocalEvents_[idx]);

            if (pendingWorkers_.fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                // Last worker to finish this frame -- wake the main thread.
                std::lock_guard<std::mutex> lock(mtx_);
                cvDone_.notify_one();
            }
        }
    }

    void EventGenerator::initialize(
        int width,
        int height)
    {
        // Tear down any existing pool before we resize state it depends on
        // (row ranges, thread count cap, per-thread buffers).
        stopPool();

        width_ = width;
        height_ = height;

        pixels_.clear();
        pixels_.resize(width_ * height_);

        numThreads_ = std::min<unsigned int>(numThreads_,
                                              static_cast<unsigned int>(std::max(1, height_)));

        threadLocalEvents_.resize(numThreads_);

        startPool();

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

    // Processes rows [yStart, yEnd). Each worker only ever touches its own
    // row range of pixels_, so there's no shared-state race between threads
    // -- no locking needed inside this function.
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

        auto t0 = std::chrono::high_resolution_clock::now();

        //------------------------------------------
        // Dispatch to the persistent pool: set the shared per-frame inputs,
        // bump generation_, wake every worker. Each worker wakes, snapshots
        // the inputs under the lock, then processes its fixed row range
        // (computed once in startPool()) with zero thread-creation cost.
        //------------------------------------------

        {
            std::lock_guard<std::mutex> lock(mtx_);
            image_ = image;
            stridePixels_ = stridePixels;
            timestamp_ = timestamp;
            positiveThreshold_ = config_.positiveThreshold;
            negativeThreshold_ = config_.negativeThreshold;
            pendingWorkers_.store(static_cast<int>(numThreads_), std::memory_order_release);
            ++generation_;
        }
        std::cerr << "[EventGenerator] dispatching to " << numThreads_ << " workers, gen="
                  << generation_ << std::endl; // TEMP DIAGNOSTIC
        cvStart_.notify_all();

        {
            std::unique_lock<std::mutex> lock(mtx_);
            cvDone_.wait(lock, [&]
                         { return pendingWorkers_.load(std::memory_order_acquire) == 0; });
        }
        std::cerr << "[EventGenerator] pool wait complete, gen=" << generation_ << std::endl; // TEMP DIAGNOSTIC

        //------------------------------------------
        // Merge. Threads own contiguous, increasing row ranges, so
        // concatenating threadLocalEvents_ in thread order reproduces the
        // same raster-scan event order the single-threaded version gave
        // (NOT strict time order in either version).
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
