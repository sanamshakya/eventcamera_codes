#pragma once

#include <cstdint>

#include "Config.h"
#include "Event.h"
#include "PixelState.h"
#include "EventPacket.h"
#include "IEventGenerator.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <utility>

namespace evsim
{

class EventGenerator : public IEventGenerator
{
public:

    explicit EventGenerator(const Config& config);

    /**
     * Initialize internal state.
     */
    void initialize(int width, int height) override;

    /**
     * Reset simulator state.
     */
    void reset() override;

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
        double timestamp) override;
        
     Config config_;

private:

    inline float intensity(uint16_t value) const;

    // Matches generate()'s pointer+stride contract - no data copy.
    void initializeFirstFrame(
        const uint16_t *image,
        int stridePixels,
        double timestamp);
        
           
    ~EventGenerator();  // add if you don't already have a destructor declared

    void startPool();
    void stopPool();
    void workerLoop(unsigned int idx);

    void processRowRange(
        const uint16_t *image,
        int stridePixels,
        double timestamp,
        int yStart,
        int yEnd,
        float positiveThreshold,
        float negativeThreshold,
        std::vector<Event> &outEvents);

    // Thread pool
    unsigned int numThreads_ = 4;
    std::vector<std::thread> pool_;
    std::vector<std::vector<Event>> threadLocalEvents_;
    std::vector<std::pair<int, int>> rowRanges_;   // per-thread [yStart, yEnd)
    std::vector<uint64_t> lastSeenGeneration_;      // unused by main thread, kept for clarity

    std::mutex mtx_;
    std::condition_variable cvStart_;
    std::condition_variable cvDone_;
    bool stop_ = false;
    uint64_t generation_ = 0;
    std::atomic<int> pendingWorkers_{0};

    // Per-frame inputs, written by generate() under mtx_, read by workers under mtx_
    const uint16_t *image_ = nullptr;
    int stridePixels_ = 0;
    double timestamp_ = 0.0;
    float positiveThreshold_ = 0.0f;
    float negativeThreshold_ = 0.0f;

private:

    

    int width_ = 0;

    int height_ = 0;

    bool initialized_ = false;

    std::vector<PixelState> pixels_;

    uint64_t frameNumber_ = 0;

};

} // namespace evsim
