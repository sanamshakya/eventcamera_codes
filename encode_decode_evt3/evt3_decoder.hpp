// evt3_decoder.hpp
// Header-only decoder for Prophesee EVT 3.0 files (ASCII header + 16-bit little-endian words).
//
//   evt3::Info info; std::vector<evt3::Event> ev; std::string err;
//   if (!evt3::load("capture.raw", info, ev, err)) { ... }
//
// Events are returned sorted by time (microseconds). The 24-bit hardware time base is unwrapped,
// so recordings longer than 16.7 s keep increasing timestamps.
// Triggers, OTHERS and CONTINUED words are skipped (counted in Info::ignored).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace evt3 {

struct Event {
    int64_t t;  // microseconds
    uint16_t x, y;
    uint8_t p;  // 1 = CD_ON, 0 = CD_OFF
};

struct Info {
    int width = 0, height = 0;  // from header, else derived from the data
    uint64_t words = 0;         // number of 16-bit data words
    uint64_t ignored = 0;       // trigger / others / continued words
};

inline bool load(const std::string& path, Info& info, std::vector<Event>& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }

    // ---- ASCII header: lines starting with '%', terminated by "% end"
    std::string line;
    while (f.peek() == '%' && std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("% evt ", 0) == 0 && line.find("3.0") == std::string::npos) {
            err = "unsupported encoding (" + line + "), only EVT 3.0 is handled";
            return false;
        }
        if (line.rfind("% format ", 0) == 0) {
            if (line.find("EVT3") == std::string::npos) {
                err = "unsupported encoding (" + line + "), only EVT 3.0 is handled";
                return false;
            }
            size_t p;
            if ((p = line.find("height=")) != std::string::npos) info.height = std::atoi(line.c_str() + p + 7);
            if ((p = line.find("width=")) != std::string::npos) info.width = std::atoi(line.c_str() + p + 6);
        }
        int w = 0, h = 0;
        if (std::sscanf(line.c_str(), "%% geometry %dx%d", &w, &h) == 2) { info.width = w; info.height = h; }
        if (line.rfind("% end", 0) == 0) break;
    }

    // ---- payload
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const size_t nWords = buf.size() / 2;
    info.words = nWords;

    uint32_t high = 0, low = 0, y = 0, baseX = 0, basePol = 0;
    int32_t lastHigh = -1;
    int64_t wrap = 0;
    int maxX = 0, maxY = 0;
    out.clear();
    out.reserve(nWords);

    auto push = [&](uint32_t x, uint32_t pol) {
        const int64_t t = wrap + ((static_cast<int64_t>(high) << 12) | low);
        out.push_back({t, static_cast<uint16_t>(x), static_cast<uint16_t>(y), static_cast<uint8_t>(pol)});
        maxX = std::max<int>(maxX, static_cast<int>(x));
        maxY = std::max<int>(maxY, static_cast<int>(y));
    };

    for (size_t i = 0; i < nWords; ++i) {
        const uint16_t w = static_cast<uint16_t>(buf[2 * i] | (buf[2 * i + 1] << 8));
        const uint32_t pl = w & 0x0FFF;
        switch (w >> 12) {
            case 0x0: y = pl & 0x7FF; break;                       // EVT_ADDR_Y
            case 0x2: push(pl & 0x7FF, (pl >> 11) & 1); break;     // EVT_ADDR_X
            case 0x3: baseX = pl & 0x7FF; basePol = (pl >> 11) & 1; break;  // VECT_BASE_X
            case 0x4:                                              // VECT_12
                for (uint32_t b = 0; b < 12; ++b)
                    if (pl & (1u << b)) push(baseX + b, basePol);
                baseX += 12;
                break;
            case 0x5:                                              // VECT_8
                for (uint32_t b = 0; b < 8; ++b)
                    if (pl & (1u << b)) push(baseX + b, basePol);
                baseX += 8;
                break;
            case 0x6: low = pl; break;                             // EVT_TIME_LOW
            case 0x8:                                              // EVT_TIME_HIGH
                if (lastHigh >= 0 && static_cast<int32_t>(pl) < lastHigh && lastHigh - static_cast<int32_t>(pl) > 2048)
                    wrap += (1LL << 24);  // 24-bit time base wrapped
                lastHigh = static_cast<int32_t>(pl);
                high = pl;
                break;
            default: ++info.ignored; break;  // EXT_TRIGGER, OTHERS, CONTINUED_*
        }
    }

    // Time low can wrap a hair before time high is updated: restore ordering if needed.
    if (!std::is_sorted(out.begin(), out.end(), [](const Event& a, const Event& b) { return a.t < b.t; }))
        std::stable_sort(out.begin(), out.end(), [](const Event& a, const Event& b) { return a.t < b.t; });

    if (info.width <= 0) info.width = maxX + 1;
    if (info.height <= 0) info.height = maxY + 1;
    return true;
}

}  // namespace evt3
