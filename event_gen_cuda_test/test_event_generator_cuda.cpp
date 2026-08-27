// Build (on the Jetson / any machine with the CUDA toolkit):
//   nvcc -O2 -std=c++17 -arch=sm_87 \
//        EventGeneratorCUDA.cu test_event_generator_cuda.cpp -o test_event_generator
//   (use -arch=sm_72 for Xavier, -arch=sm_87 for Orin)
//
// Run:
//   ./test_event_generator
// Exit code is 0 if every test passes, 1 otherwise -- wire this into CI /
// CTest by just checking the process exit code.

#include "EventGeneratorCUDA.cuh"
#include "EventGeneratorRef.h"

#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <cstdio>
#include <cmath>

using namespace evsim;

namespace
{

std::vector<uint16_t> BuildFrame(int width, int height,
                                  const std::function<uint16_t(int, int)> &pixelFn)
{
    std::vector<uint16_t> frame(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            frame[static_cast<size_t>(y) * width + x] = pixelFn(x, y);
    return frame;
}

// Canonical ordering so the GPU's atomic-reservation order and the CPU's
// raster-scan order can be compared regardless of the order events were
// actually produced in. Within one pixel, crossing order is preserved by
// increasing timestamp, so (y, x, timestamp) is a stable sort key.
void SortEvents(std::vector<Event> &events)
{
    std::sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
        if (a.y != b.y) return a.y < b.y;
        if (a.x != b.x) return a.x < b.x;
        return a.timestamp < b.timestamp;
    });
}

bool CompareEvents(std::vector<Event> golden, std::vector<Event> actual,
                    double timeTolerance, int maxMismatchesToPrint)
{
    SortEvents(golden);
    SortEvents(actual);

    if (golden.size() != actual.size())
    {
        printf("  MISMATCH: golden has %zu events, GPU has %zu events\n",
               golden.size(), actual.size());
    }

    const size_t n = std::min(golden.size(), actual.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const Event &g = golden[i];
        const Event &a = actual[i];
        const bool ok = (g.x == a.x) && (g.y == a.y) && (g.polarity == a.polarity) &&
                        (std::fabs(g.timestamp - a.timestamp) <= timeTolerance);
        if (!ok)
        {
            if (mismatches < static_cast<size_t>(maxMismatchesToPrint))
            {
                printf("  [%zu] golden=(x=%u,y=%u,t=%.9f,p=%d)  gpu=(x=%u,y=%u,t=%.9f,p=%d)\n",
                       i, g.x, g.y, g.timestamp, static_cast<int>(g.polarity),
                       a.x, a.y, a.timestamp, static_cast<int>(a.polarity));
            }
            ++mismatches;
        }
    }
    if (mismatches > 0)
        printf("  %zu / %zu compared events mismatched\n", mismatches, n);

    return (golden.size() == actual.size()) && (mismatches == 0);
}

} // namespace

// ---------------------------------------------------------------------
// Test 1: Small, hand-checkable pattern. Every region's expected event
// count is computed by hand below, so this test also catches the golden
// model itself being wrong, not just the GPU vs. CPU divergence.
// ---------------------------------------------------------------------
bool RunDeterministicTest(const Config &cfg)
{
    printf("\n=== Test 1: Deterministic pattern ===\n");

    const int width = 64, height = 48;
    const uint16_t baseline = 1000;

    // [0,20)x[0,20)  : +1x  positiveThreshold -> exactly 1 ON event/pixel
    // [20,40)x[0,20) : +3.5x positiveThreshold -> exactly 3 ON events/pixel (floor(3.5)=3)
    // [40,60)x[0,20) : -2x  negativeThreshold  -> exactly 2 OFF events/pixel
    // everything else: unchanged -> 0 events/pixel
    auto frame0 = BuildFrame(width, height, [&](int, int) { return baseline; });
    auto frame1 = BuildFrame(width, height, [&](int x, int y) -> uint16_t {
        float delta = 0.0f;
        if (y < 20 && x < 20)                  delta = 1.0f * cfg.positiveThreshold;
        else if (y < 20 && x >= 20 && x < 40)   delta = 3.5f * cfg.positiveThreshold;
        else if (y < 20 && x >= 40 && x < 60)   delta = 2.0f * cfg.negativeThreshold;
        return static_cast<uint16_t>(std::max(0.0f, static_cast<float>(baseline) + delta));
    });

    const size_t expectedEvents =
        static_cast<size_t>(20 * 20) * 1 +
        static_cast<size_t>(20 * 20) * 3 +
        static_cast<size_t>(20 * 20) * 2;

    EventGeneratorRef golden(cfg.positiveThreshold, cfg.negativeThreshold);
    EventGeneratorCUDA gpu(cfg, /*maxEventsPerFrame=*/static_cast<int>(expectedEvents) * 2);

    golden.initialize(width, height);
    gpu.initialize(width, height);

    auto g0 = golden.generate(frame0.data(), width, 0.0);
    auto d0 = gpu.generate(frame0.data(), width, 0.0);
    const bool firstFrameOk = g0.events.empty() && d0.events.empty();
    printf("First-frame init: golden events=%zu gpu events=%zu %s\n",
           g0.events.size(), d0.events.size(), firstFrameOk ? "[OK]" : "[FAIL]");

    auto g1 = golden.generate(frame1.data(), width, 1.0 / 30.0);
    auto d1 = gpu.generate(frame1.data(), width, 1.0 / 30.0);

    printf("Expected events (hand-computed): %zu\n", expectedEvents);
    printf("Golden CPU produced:             %zu\n", g1.events.size());
    printf("GPU produced:                    %zu\n", d1.events.size());

    const bool countOk = (g1.events.size() == expectedEvents);
    const bool matchOk = CompareEvents(g1.events, d1.events, /*timeTolerance=*/1e-6, /*maxPrint=*/20);

    const bool pass = firstFrameOk && countOk && matchOk;
    printf("Test 1 result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ---------------------------------------------------------------------
// Test 2: Randomized multi-frame sequence at real sensor resolution.
// Broader coverage than Test 1 -- random per-pixel walks, both positive
// and negative crossings, occasional multi-crossing jumps.
// ---------------------------------------------------------------------
bool RunRandomizedTest(const Config &cfg, int width, int height, int numFrames, unsigned seed)
{
    printf("\n=== Test 2: Randomized sequence (%dx%d, %d frames) ===\n", width, height, numFrames);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> baseDist(500, 3500);
    // Real event-camera scenes are sparse: only a small fraction of pixels
    // change per frame. ~2% active, magnitude in [threshold, 4x threshold]
    // so every active pixel is guaranteed to fire 1-4 events -- this keeps
    // the per-frame event count realistic (tens of thousands, not millions)
    // so it stays comfortably under maxEventsPerFrame below.
    std::bernoulli_distribution activeDist(0.02);
    std::uniform_int_distribution<int> magDist(
        static_cast<int>(cfg.positiveThreshold), static_cast<int>(cfg.positiveThreshold) * 4);
    std::bernoulli_distribution signDist(0.5);

    std::vector<uint16_t> current(static_cast<size_t>(width) * height);
    for (auto &v : current) v = static_cast<uint16_t>(baseDist(rng));

    EventGeneratorRef golden(cfg.positiveThreshold, cfg.negativeThreshold);
    EventGeneratorCUDA gpu(cfg, /*maxEventsPerFrame=*/200000);

    golden.initialize(width, height);
    gpu.initialize(width, height);
    golden.generate(current.data(), width, 0.0);
    gpu.generate(current.data(), width, 0.0);

    bool allOk = true;
    for (int f = 1; f <= numFrames; ++f)
    {
        for (auto &v : current)
        {
            if (!activeDist(rng)) continue; // most pixels: unchanged this frame
            const int mag = magDist(rng);
            const int step = signDist(rng) ? mag : -mag;
            const int nv = static_cast<int>(v) + step;
            v = static_cast<uint16_t>(std::clamp(nv, 0, 4095));
        }
        const double t = f / 30.0;

        auto g = golden.generate(current.data(), width, t);
        auto d = gpu.generate(current.data(), width, t);

        const bool ok = CompareEvents(g.events, d.events, 1e-6, 5);
        printf("  frame %3d: golden=%6zu gpu=%6zu %s\n",
               f, g.events.size(), d.events.size(), ok ? "[OK]" : "[FAIL]");
        allOk = allOk && ok;
    }

    printf("Test 2 result: %s\n", allOk ? "PASS" : "FAIL");
    return allOk;
}

// ---------------------------------------------------------------------
// Test 3: Deliberately undersized event capacity -- verifies the
// overflow guard clips instead of overflowing/crashing.
// ---------------------------------------------------------------------
bool RunOverflowTest(const Config &cfg)
{
    printf("\n=== Test 3: Event-buffer overflow handling ===\n");

    const int width = 32, height = 32; // 1024 pixels
    const int smallCap = 100;           // deliberately smaller than the frame will produce

    auto frame0 = BuildFrame(width, height, [](int, int) { return static_cast<uint16_t>(1000); });
    // every pixel jumps 5x positiveThreshold -> 5 events/pixel -> 5120 events, far over cap
    auto frame1 = BuildFrame(width, height, [&](int, int) {
        return static_cast<uint16_t>(1000 + 5 * cfg.positiveThreshold);
    });

    EventGeneratorCUDA gpu(cfg, smallCap);
    gpu.initialize(width, height);
    gpu.generate(frame0.data(), width, 0.0);
    auto d1 = gpu.generate(frame1.data(), width, 1.0 / 30.0);

    const bool ok = (d1.events.size() == static_cast<size_t>(smallCap));
    printf("Capacity=%d, events returned=%zu %s (an overflow warning should print above)\n",
           smallCap, d1.events.size(), ok ? "[OK]" : "[FAIL]");
    printf("Test 3 result: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main()
{
    Config cfg{};
    cfg.positiveThreshold = 15.0f;
    cfg.negativeThreshold = -15.0f;

    const bool t1 = RunDeterministicTest(cfg);
    const bool t2 = RunRandomizedTest(cfg, 1280, 720, /*numFrames=*/20, /*seed=*/42);
    const bool t3 = RunOverflowTest(cfg);

    printf("\n=== Summary ===\n");
    printf("Test 1 (deterministic): %s\n", t1 ? "PASS" : "FAIL");
    printf("Test 2 (randomized):    %s\n", t2 ? "PASS" : "FAIL");
    printf("Test 3 (overflow):      %s\n", t3 ? "PASS" : "FAIL");

    return (t1 && t2 && t3) ? 0 : 1;
}
