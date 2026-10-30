// EventLogger.h
//
// Same public API and the same CSV format as before
// (<path>.events.csv: label,frame,t,x,y,polarity ; <path>.frames.csv:
// label,frame,t,event_count), so compare_events.py is unaffected.
//
// What changed: rows are formatted by hand straight into one large char
// buffer and written with a single fwrite per ~4 MB. The old version did
// one fprintf("%s,%d,%.6f,...") per event (slow at millions of events/s)
// and an fflush every 50k events.
//
// NOT thread-safe: use from one thread only (the logger thread in main.cpp).

#pragma once

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace evsim
{

class EventLogger
{
public:
    EventLogger(const std::string& path, const std::string& label);
    ~EventLogger();

    // t is the EVENT's own timestamp in seconds (written with 6 decimals,
    // i.e. microsecond resolution, same as before).
    void log(int frameIndex, double t, int x, int y, int polarity);

    // One row per frame, including zero-event frames.
    void logFrameCount(int frameIndex, double t, int eventCount);

    // Write everything buffered and fflush both files.
    void flush();

private:
    void drain(); // fwrite the buffer, no fflush

    std::string label_;
    FILE* eventsFile_ = nullptr;
    FILE* framesFile_ = nullptr;

    std::vector<char> out_;
    size_t used_ = 0;
    static constexpr size_t kBufBytes = size_t(1) << 22; // 4 MiB
};

} // namespace evsim
