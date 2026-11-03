#include "EventLogger.h"

#include <cstdint>
#include <cstring>

using namespace evsim;

namespace
{

inline char* appendUInt(char* p, unsigned long long v)
{
    char tmp[24];
    int n = 0;
    do { tmp[n++] = static_cast<char>('0' + v % 10); v /= 10; } while (v);
    while (n) *p++ = tmp[--n];
    return p;
}

inline char* appendInt(char* p, long long v)
{
    if (v < 0)
    {
        *p++ = '-';
        return appendUInt(p, 0ULL - static_cast<unsigned long long>(v));
    }
    return appendUInt(p, static_cast<unsigned long long>(v));
}

// Equivalent of "%.6f" for sane values (rounds the microsecond count
// half-up; printf rounds the exact binary value, so the very last digit can
// differ in rare tie cases -- irrelevant at microsecond resolution).
inline char* appendFixed6(char* p, double t)
{
    if (t < 0.0) { *p++ = '-'; t = -t; }
    const unsigned long long us = static_cast<unsigned long long>(t * 1e6 + 0.5);
    p = appendUInt(p, us / 1000000ULL);
    *p++ = '.';
    const unsigned frac = static_cast<unsigned>(us % 1000000ULL);
    for (unsigned d = 100000; d > 0; d /= 10)
        *p++ = static_cast<char>('0' + (frac / d) % 10);
    return p;
}

} // namespace

EventLogger::EventLogger(const std::string& path, const std::string& label)
    : label_(label)
{
    eventsFile_ = std::fopen((path + ".events.csv").c_str(), "w");
    framesFile_ = std::fopen((path + ".frames.csv").c_str(), "w");

    if (eventsFile_)
        std::fprintf(eventsFile_, "label,frame,t,x,y,polarity\n");
    if (framesFile_)
        std::fprintf(framesFile_, "label,frame,t,event_count\n");

    out_.resize(kBufBytes + 128 + label_.size());
}

EventLogger::~EventLogger()
{
    flush();
    if (eventsFile_) std::fclose(eventsFile_);
    if (framesFile_) std::fclose(framesFile_);
}

void EventLogger::log(int frameIndex, double t, int x, int y, int polarity)
{
    if (!eventsFile_) return;

    // Worst-case row: label + 3 ints + one time (~25 chars) + separators.
    char* p = out_.data() + used_;
    std::memcpy(p, label_.data(), label_.size());
    p += label_.size();
    *p++ = ',';
    p = appendInt(p, frameIndex);
    *p++ = ',';
    p = appendFixed6(p, t);
    *p++ = ',';
    p = appendInt(p, x);
    *p++ = ',';
    p = appendInt(p, y);
    *p++ = ',';
    p = appendInt(p, polarity);
    *p++ = '\n';
    used_ = static_cast<size_t>(p - out_.data());

    if (used_ >= kBufBytes)
        drain();
}

void EventLogger::logFrameCount(int frameIndex, double t, int eventCount)
{
    if (!framesFile_) return;
    std::fprintf(framesFile_, "%s,%d,%.6f,%d\n",
                 label_.c_str(), frameIndex, t, eventCount);
}

void EventLogger::drain()
{
    if (eventsFile_ && used_)
        std::fwrite(out_.data(), 1, used_, eventsFile_);
    used_ = 0;
}

void EventLogger::flush()
{
    drain();
    if (eventsFile_) std::fflush(eventsFile_);
    if (framesFile_) std::fflush(framesFile_);
}
