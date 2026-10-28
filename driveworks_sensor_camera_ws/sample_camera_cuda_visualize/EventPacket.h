#pragma once

#include <vector>

#include "Event.h"

namespace evsim
{

struct EventPacket
{
    std::vector<Event> events;

    double startTime = 0.0;
    double endTime = 0.0;

    uint64_t frameNumber = 0;

    void clear()
    {
        events.clear();
    }

    size_t size() const
    {
        return events.size();
    }

    bool empty() const
    {
        return events.empty();
    }
};

} // namespace evsim