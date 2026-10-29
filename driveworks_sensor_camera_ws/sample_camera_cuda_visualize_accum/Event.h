#pragma once

#include <cstdint>

namespace evsim
{

    struct Event
    {
        uint16_t x;
        uint16_t y;

        double timestamp;

        int8_t polarity;
    };

}