// evt3_encoder.cpp
// Encode an event capture stored in CSV into the Prophesee EVT 3.0 format.
//
// CSV columns (header required, order free): label,frame,t,x,y,polarity
//   polarity: >0 -> CD_ON (1), otherwise CD_OFF (0)   (-1 / +1 and 0 / 1 both work)
//
// Build:  g++ -O2 -std=c++17 -o evt3_encoder evt3_encoder.cpp
// Usage:  ./evt3_encoder events.csv out.raw [--time-unit s|ms|us] [--frame-period P]
//                                        [--width W] [--height H] [--verify]
//
// EVT 3.0 summary (16-bit little-endian words, type in the 4 MSBs):
//   0x0 EVT_ADDR_Y   [11]=system_type(0) [10:0]=y
//   0x2 EVT_ADDR_X   [11]=pol [10:0]=x          single event
//   0x3 VECT_BASE_X  [11]=pol [10:0]=x          base for following vectors
//   0x4 VECT_12      [11:0]=validity bits       events at base+i, base += 12
//   0x5 VECT_8       [7:0]=validity bits        (decoded by --verify, not emitted)
//   0x6 EVT_TIME_LOW [11:0]=t[11:0]
//   0x8 EVT_TIME_HIGH[11:0]=t[23:12]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

struct Ev {
    int64_t t;  // microseconds
    uint16_t x, y;
    uint8_t p;  // 0 = OFF, 1 = ON
};

enum : uint16_t {
    EVT_ADDR_Y = 0x0,
    EVT_ADDR_X = 0x2,
    VECT_BASE_X = 0x3,
    VECT_12 = 0x4,
    VECT_8 = 0x5,
    EVT_TIME_LOW = 0x6,
    EVT_TIME_HIGH = 0x8,
};

static inline uint16_t makeWord(uint16_t type, uint16_t payload12) {
    return static_cast<uint16_t>((type << 12) | (payload12 & 0x0FFF));
}

// ---------------------------------------------------------------- encoder
class Evt3Encoder {
public:
    std::vector<uint16_t> words;

    // Events are sorted in place by (t, y, pol, x).
    void encode(std::vector<Ev>& evs) {
        std::sort(evs.begin(), evs.end(), [](const Ev& a, const Ev& b) {
            return std::tie(a.t, a.y, a.p, a.x) < std::tie(b.t, b.y, b.p, b.x);
        });
        const size_t n = evs.size();
        size_t i = 0;
        while (i < n) {
            size_t k = i;
            while (k < n && evs[k].t == evs[i].t) ++k;  // block with identical timestamp
            emitTime(evs[i].t);
            size_t g = i;
            while (g < k) {  // sub-block with identical (y, pol)
                size_t h = g;
                while (h < k && evs[h].y == evs[g].y && evs[h].p == evs[g].p) ++h;
                emitRow(evs, g, h);
                g = h;
            }
            i = k;
        }
    }

private:
    int32_t lastHigh_ = -1, lastLow_ = -1, lastY_ = -1;

    void emitTime(int64_t t) {
        const int32_t high = static_cast<int32_t>((t >> 12) & 0xFFF);
        const int32_t low = static_cast<int32_t>(t & 0xFFF);
        if (high != lastHigh_) {
            words.push_back(makeWord(EVT_TIME_HIGH, high));
            words.push_back(makeWord(EVT_TIME_LOW, low));  // always re-send low after a new high
            lastHigh_ = high;
            lastLow_ = low;
        } else if (low != lastLow_) {
            words.push_back(makeWord(EVT_TIME_LOW, low));
            lastLow_ = low;
        }
    }

    // All events in [g,h) share timestamp, y and polarity and are sorted by x.
    void emitRow(const std::vector<Ev>& evs, size_t g, size_t h) {
        const uint16_t y = evs[g].y;
        const uint16_t pol = evs[g].p;
        if (y != lastY_) {
            words.push_back(makeWord(EVT_ADDR_Y, y & 0x7FF));  // system_type = 0 (master)
            lastY_ = y;
        }

        std::vector<uint16_t> xs, dups;  // vectors can't hold two events on the same pixel
        for (size_t k = g; k < h; ++k) {
            if (!xs.empty() && xs.back() == evs[k].x) dups.push_back(evs[k].x);
            else xs.push_back(evs[k].x);
        }

        const size_t n = xs.size();
        size_t i = 0;
        while (i < n) {
            const uint32_t base = xs[i];
            std::vector<uint16_t> chunks;  // each chunk = one VECT_12 word
            size_t j = i;
            // keep chaining while the next event lands in the *next* 12-wide window
            while (j < n && xs[j] < base + 12 * (chunks.size() + 1)) {
                const uint32_t lo = base + 12 * static_cast<uint32_t>(chunks.size());
                uint16_t mask = 0;
                while (j < n && xs[j] < lo + 12) mask |= static_cast<uint16_t>(1u << (xs[j++] - lo));
                chunks.push_back(mask);
            }
            if (chunks.size() == 1 && (chunks[0] & (chunks[0] - 1)) == 0) {
                singleEvent(base, pol);  // lone event: 1 word beats BASE_X + VECT_12
            } else {
                words.push_back(makeWord(VECT_BASE_X, static_cast<uint16_t>((pol << 11) | (base & 0x7FF))));
                for (uint16_t m : chunks) words.push_back(makeWord(VECT_12, m));
            }
            i = j;
        }
        for (uint16_t x : dups) singleEvent(x, pol);
    }

    void singleEvent(uint32_t x, uint16_t pol) {
        words.push_back(makeWord(EVT_ADDR_X, static_cast<uint16_t>((pol << 11) | (x & 0x7FF))));
    }
};

// ---------------------------------------------------------------- decoder (for --verify)
static bool decodeFile(const std::string& path, std::vector<Ev>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("% end", 0) == 0) break;
        if (line.empty() || line[0] != '%') return false;
    }
    uint32_t high = 0, low = 0, y = 0, baseX = 0, basePol = 0;
    unsigned char b[2];
    while (f.read(reinterpret_cast<char*>(b), 2)) {
        const uint16_t w = static_cast<uint16_t>(b[0] | (b[1] << 8));
        const uint16_t type = w >> 12, pl = w & 0x0FFF;
        const int64_t t = (static_cast<int64_t>(high) << 12) | low;
        switch (type) {
            case EVT_ADDR_Y: y = pl & 0x7FF; break;
            case EVT_ADDR_X: out.push_back({t, static_cast<uint16_t>(pl & 0x7FF), static_cast<uint16_t>(y), static_cast<uint8_t>(pl >> 11)}); break;
            case VECT_BASE_X: baseX = pl & 0x7FF; basePol = pl >> 11; break;
            case VECT_12:
                for (int i = 0; i < 12; ++i)
                    if (pl & (1u << i)) out.push_back({t, static_cast<uint16_t>(baseX + i), static_cast<uint16_t>(y), static_cast<uint8_t>(basePol)});
                baseX += 12;
                break;
            case VECT_8:
                for (int i = 0; i < 8; ++i)
                    if (pl & (1u << i)) out.push_back({t, static_cast<uint16_t>(baseX + i), static_cast<uint16_t>(y), static_cast<uint8_t>(basePol)});
                baseX += 8;
                break;
            case EVT_TIME_LOW: low = pl; break;
            case EVT_TIME_HIGH: high = pl; break;
            default: break;  // triggers / others / continued: ignored
        }
    }
    return true;
}

// ---------------------------------------------------------------- CSV
static std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> r;
    std::string cur;
    std::stringstream ss(s);
    while (std::getline(ss, cur, d)) r.push_back(cur);
    return r;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " in.csv out.raw [--time-unit s|ms|us] [--frame-period P] [--width W] [--height H] [--verify]\n";
        return 1;
    }
    const std::string inPath = argv[1], outPath = argv[2];
    double scale = 1e3;  // default: CSV time is in milliseconds -> us
    double framePeriod = 0.0;
    int width = 0, height = 0;
    bool verify = false;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--time-unit" && i + 1 < argc) {
            const std::string u = argv[++i];
            scale = (u == "s") ? 1e6 : (u == "ms") ? 1e3 : (u == "us") ? 1.0 : -1.0;
            if (scale < 0) { std::cerr << "time-unit must be s, ms or us\n"; return 1; }
        } else if (a == "--frame-period" && i + 1 < argc) framePeriod = std::atof(argv[++i]);
        else if (a == "--width" && i + 1 < argc) width = std::atoi(argv[++i]);
        else if (a == "--height" && i + 1 < argc) height = std::atoi(argv[++i]);
        else if (a == "--verify") verify = true;
        else { std::cerr << "Unknown argument: " << a << "\n"; return 1; }
    }

    std::ifstream in(inPath);
    if (!in) { std::cerr << "Cannot open " << inPath << "\n"; return 1; }

    std::string line;
    std::getline(in, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto cols = split(line, ',');
    auto find = [&](const char* name) {
        for (size_t i = 0; i < cols.size(); ++i) if (cols[i] == name) return static_cast<int>(i);
        return -1;
    };
    const int cFrame = find("frame"), cT = find("t"), cX = find("x"), cY = find("y"), cP = find("polarity");
    if (cT < 0 || cX < 0 || cY < 0 || cP < 0) {
        std::cerr << "CSV header must contain: t,x,y,polarity (frame optional)\n";
        return 1;
    }
    const size_t need = static_cast<size_t>(std::max({cT, cX, cY, cP, cFrame})) + 1;

    std::vector<Ev> evs;
    size_t bad = 0, lineNo = 1;
    int maxX = 0, maxY = 0;
    bool wrapWarn = false;
    while (std::getline(in, line)) {
        ++lineNo;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto f = split(line, ',');
        if (f.size() < need) { ++bad; continue; }
        try {
            double t = std::stod(f[cT]);
            if (cFrame >= 0 && framePeriod != 0.0) t += (std::stol(f[cFrame]) - 1) * framePeriod;
            const int64_t tus = std::llround(t * scale);
            const int x = std::stoi(f[cX]), y = std::stoi(f[cY]);
            const int pol = std::stoi(f[cP]);
            if (tus < 0 || x < 0 || y < 0 || x > 2047 || y > 2047) { ++bad; continue; }  // EVT3: 11-bit coords
            if (tus >= (1LL << 24)) wrapWarn = true;
            evs.push_back({tus, static_cast<uint16_t>(x), static_cast<uint16_t>(y), static_cast<uint8_t>(pol > 0 ? 1 : 0)});
            maxX = std::max(maxX, x);
            maxY = std::max(maxY, y);
        } catch (...) { ++bad; }
    }
    if (evs.empty()) { std::cerr << "No valid events read.\n"; return 1; }
    if (bad) std::cerr << "Warning: skipped " << bad << " invalid rows\n";
    if (wrapWarn)
        std::cerr << "Note: timestamps exceed 24 bits (16.7 s); TIME_HIGH wraps like on real hardware.\n";
    if (width <= 0) width = maxX + 1;
    if (height <= 0) height = maxY + 1;

    std::vector<Ev> original = evs;  // keep for verification
    Evt3Encoder enc;
    enc.encode(evs);

    std::ofstream out(outPath, std::ios::binary);
    if (!out) { std::cerr << "Cannot write " << outPath << "\n"; return 1; }
    std::ostringstream hdr;
    hdr << "% evt 3.0\n"
        << "% format EVT3;height=" << height << ";width=" << width << "\n"
        << "% geometry " << width << "x" << height << "\n"
        << "% end\n";
    const std::string h = hdr.str();
    out.write(h.data(), static_cast<std::streamsize>(h.size()));
    for (uint16_t w : enc.words) {  // explicit little-endian
        const unsigned char b[2] = {static_cast<unsigned char>(w & 0xFF), static_cast<unsigned char>(w >> 8)};
        out.write(reinterpret_cast<const char*>(b), 2);
    }
    out.close();

    std::cout << "Wrote " << outPath << ": " << original.size() << " events -> " << enc.words.size() << " words ("
              << enc.words.size() * 2 << " bytes), sensor " << width << "x" << height << "\n";

    if (verify) {
        std::vector<Ev> dec;
        if (!decodeFile(outPath, dec)) { std::cerr << "VERIFY: could not read back file\n"; return 2; }
        auto key = [](const Ev& e) { return std::make_tuple(e.t, e.y, e.x, e.p); };
        std::vector<Ev> ref = original;
        for (auto& e : ref) e.t &= 0xFFFFFF;  // 24-bit time base
        auto less = [&](const Ev& a, const Ev& b) { return key(a) < key(b); };
        std::sort(ref.begin(), ref.end(), less);
        std::sort(dec.begin(), dec.end(), less);
        bool ok = ref.size() == dec.size();
        for (size_t i = 0; ok && i < ref.size(); ++i) ok = key(ref[i]) == key(dec[i]);
        std::cout << (ok ? "VERIFY: OK (decoded stream matches CSV)\n" : "VERIFY: MISMATCH\n");
        return ok ? 0 : 2;
    }
    return 0;
}
