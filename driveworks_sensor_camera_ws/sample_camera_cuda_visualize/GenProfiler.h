#pragma once
// Lightweight per-stage profiler for EventGeneratorCUDA::generate().
// GPU-side stages use cudaEvents (accurate, measured on the stream);
// host-side stages use steady_clock. Prints one summary line block every
// `reportEvery` frames. First `warmup` frames are ignored (context/JIT/page
// faults make them unrepresentative).

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>

namespace evsim
{

struct Stat
{
    double sum = 0, mx = 0;
    long n = 0;
    void add(double v) { sum += v; mx = std::max(mx, v); ++n; }
    double avg() const { return n ? sum / n : 0.0; }
    void reset() { *this = Stat(); }
};

inline double nowMs()
{
    using C = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(C::now().time_since_epoch()).count();
}

class GenProfiler
{
public:
    explicit GenProfiler(int reportEvery = 100, int warmup = 5)
        : every_(reportEvery), warmup_(warmup)
    {
        for (auto &e : ev) cudaEventCreate(&e);
    }
    ~GenProfiler() { for (auto &e : ev) cudaEventDestroy(e); }

    // ev[0] start, ev[1] after H2D, ev[2] after kernel, ev[3] after D2H
    cudaEvent_t ev[4];

    // Host-side stage timings (ms), filled in by generate()
    double hostStage = 0;   // memcpy into pinned staging buffer
    double hostSync = 0;    // time blocked in cudaStreamSynchronize
    double hostAssign = 0;  // packet.events.assign()
    double hostTotal = 0;   // whole generate()

    // Call after cudaStreamSynchronize.
    void endFrame(size_t eventCount, size_t d2hBytes, size_t h2dBytes)
    {
        if (++frame_ <= warmup_) return;

        float h2dMs = 0, kMs = 0, d2hMs = 0;
        cudaEventElapsedTime(&h2dMs, ev[0], ev[1]);
        cudaEventElapsedTime(&kMs, ev[1], ev[2]);
        cudaEventElapsedTime(&d2hMs, ev[2], ev[3]);

        sH2D.add(h2dMs); sKernel.add(kMs); sD2H.add(d2hMs);
        sStage.add(hostStage); sSync.add(hostSync);
        sAssign.add(hostAssign); sTotal.add(hostTotal);
        sEvents.add(static_cast<double>(eventCount));
        d2hBytes_ = d2hBytes; h2dBytes_ = h2dBytes;

        if (sTotal.n >= every_) { report(); reset(); }
    }

private:
    void line(const char *name, const Stat &s) const
    {
        fprintf(stderr, "  %-22s avg %8.3f  max %8.3f ms\n", name, s.avg(), s.mx);
    }
    void report() const
    {
        const double total = sTotal.avg();
        fprintf(stderr, "[gen-prof] %ld frames, %.0f events/frame avg\n", sTotal.n, sEvents.avg());
        line("host stage memcpy", sStage);
        line("H2D (gpu)", sH2D);
        line("kernel (gpu)", sKernel);
        line("D2H (gpu)", sD2H);
        line("host blocked in sync", sSync);
        line("packet.assign", sAssign);
        line("generate() TOTAL", sTotal);
        const double d2hGBs = sD2H.avg() > 0 ? (d2hBytes_ / 1e9) / (sD2H.avg() / 1e3) : 0;
        const double h2dGBs = sH2D.avg() > 0 ? (h2dBytes_ / 1e9) / (sH2D.avg() / 1e3) : 0;
        fprintf(stderr, "  D2H %.2f MB @ %.2f GB/s | H2D %.2f MB @ %.2f GB/s\n",
                d2hBytes_ / 1e6, d2hGBs, h2dBytes_ / 1e6, h2dGBs);
        if (total > 0)
            fprintf(stderr, "  share of total: H2D %.0f%%  kernel %.0f%%  D2H %.0f%%  stage %.0f%%  assign %.0f%%\n",
                    100 * sH2D.avg() / total, 100 * sKernel.avg() / total,
                    100 * sD2H.avg() / total, 100 * sStage.avg() / total,
                    100 * sAssign.avg() / total);
    }
    void reset()
    {
        sH2D.reset(); sKernel.reset(); sD2H.reset(); sStage.reset();
        sSync.reset(); sAssign.reset(); sTotal.reset(); sEvents.reset();
    }

    int every_, warmup_;
    long frame_ = 0;
    size_t d2hBytes_ = 0, h2dBytes_ = 0;
    Stat sH2D, sKernel, sD2H, sStage, sSync, sAssign, sTotal, sEvents;
};

} // namespace evsim
