#pragma once
#include <cstdint>
#include <vector>

// Placeholder -- replace this file with your project's real EventPacket.h
// if the field names/order differ. EventGeneratorCUDA.cu and the test
// harness both construct these via aggregate initialization
// (Event{x, y, timestamp, polarity}), so field ORDER matters as much as
// names if you swap this out.

namespace evsim
{

struct Event
{
    uint16_t x;
    uint16_t y;
    double   timestamp;
    int8_t   polarity; // +1 = ON, -1 = OFF
};

struct EventPacket
{
    double   startTime   = 0.0;
    double   endTime     = 0.0;
    uint64_t frameNumber = 0;
    std::vector<Event> events;
};

} // namespace evsim
