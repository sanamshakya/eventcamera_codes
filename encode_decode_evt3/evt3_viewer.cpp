// evt3_viewer.cpp
// Decode an EVT 3.0 file and visualise it as accumulated event frames.
// The accumulation window can be changed while the viewer is running.
//
//   Interactive (needs OpenCV):   ./evt3_viewer out.raw [--accum-us 5000]
//   Headless frame dump:          ./evt3_viewer out.raw --dump frames --accum-us 100,1000,10000
//   Decode to CSV:                ./evt3_viewer out.raw --csv decoded.csv --no-gui
//
// A frame shown at time T contains the events with T - accum <= t < T; for every pixel the most
// recent polarity in that window is drawn (ON = blue, OFF = red).
//
// Interactive controls (window trackbars, plus keys):
//   trackbar "accum"     1 us .. 10 s, logarithmic        keys  + / -   double / halve
//   trackbar "speed"     0.0001x .. 100x real time        keys  ] / [   faster / slower
//   trackbar "position"  scrub through the recording      key   r       restart
//   space pause/resume   a / d  step back / forward by one window (any time, handy when paused)
//   l toggle loop        p save PNG snapshot              q / Esc quit

#include "evt3_decoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace fs = std::filesystem;

// ------------------------------------------------------------------ rendering
struct Palette {
    uint8_t bg[3], on[3], off[3];  // BGR
};
static const Palette kDark = {{30, 30, 30}, {255, 170, 60}, {60, 80, 255}};
static const Palette kLight = {{255, 255, 255}, {200, 90, 10}, {30, 30, 210}};

// Draw the events with tEnd-accum <= t < tEnd into a BGR image; returns how many were in the window.
static size_t renderFrame(const std::vector<evt3::Event>& ev, int64_t tEnd, int64_t accum, int W, int H,
                          const Palette& pal, std::vector<uint8_t>& img) {
    img.resize(static_cast<size_t>(W) * H * 3);
    for (size_t i = 0; i < img.size(); i += 3) {
        img[i] = pal.bg[0];
        img[i + 1] = pal.bg[1];
        img[i + 2] = pal.bg[2];
    }
    auto before = [](const evt3::Event& e, int64_t t) { return e.t < t; };
    const auto lo = std::lower_bound(ev.begin(), ev.end(), tEnd - accum, before);
    const auto hi = std::lower_bound(lo, ev.end(), tEnd, before);
    for (auto it = lo; it != hi; ++it) {
        if (it->x >= W || it->y >= H) continue;
        uint8_t* px = &img[(static_cast<size_t>(it->y) * W + it->x) * 3];
        const uint8_t* c = it->p ? pal.on : pal.off;
        px[0] = c[0];
        px[1] = c[1];
        px[2] = c[2];
    }
    return static_cast<size_t>(hi - lo);
}

static std::string fmtTime(double us) {
    char b[48];
    if (us < 1000.0) std::snprintf(b, sizeof b, "%.0f us", us);
    else if (us < 1e6) std::snprintf(b, sizeof b, "%.3f ms", us / 1e3);
    else std::snprintf(b, sizeof b, "%.3f s", us / 1e6);
    return b;
}

// Writes <base>.png with OpenCV, or <base>.ppm without it. Returns the path written.
static std::string saveImage(const std::string& base, const std::vector<uint8_t>& bgr, int W, int H) {
#ifdef HAVE_OPENCV
    cv::Mat m(H, W, CV_8UC3, const_cast<uint8_t*>(bgr.data()));
    const std::string path = base + ".png";
    cv::imwrite(path, m);
    return path;
#else
    const std::string path = base + ".ppm";
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";
    std::vector<uint8_t> rgb(bgr.size());
    for (size_t i = 0; i < bgr.size(); i += 3) {
        rgb[i] = bgr[i + 2];
        rgb[i + 1] = bgr[i + 1];
        rgb[i + 2] = bgr[i];
    }
    f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return path;
#endif
}

// ------------------------------------------------------------------ CLI
struct Args {
    std::string file;
    std::vector<int64_t> accum;  // microseconds (several values allowed with --dump)
    double speed = -1.0;         // x real time; <0 = auto
    int fps = 30;
    int scale = 0;               // 0 = auto
    bool light = false, noGui = false, noLoop = false;
    std::string dumpDir, csv;
    int64_t step = 0;            // dump: time between frames, 0 = same as accum
    int maxFrames = 1000;
};

static void usage(const char* a) {
    std::cerr << "Usage: " << a << " file.raw [options]\n"
              << "  --accum-us N[,N...]  accumulation window(s) in microseconds (default: auto)\n"
              << "  --speed X            playback speed, 1 = real time (default: whole file in ~10 s)\n"
              << "  --fps N              display refresh rate (default 30)\n"
              << "  --scale N            integer upscaling of the window (default: auto)\n"
              << "  --light              white background\n"
              << "  --no-loop            stop at the end instead of looping\n"
              << "  --dump DIR           headless: write frames to DIR (PNG with OpenCV, else PPM)\n"
              << "  --step-us N          dump: time between frames (default: = accum)\n"
              << "  --max-frames N       dump: frame limit per accumulation value (default 1000)\n"
              << "  --csv FILE           write decoded events as CSV (t in us, polarity 1/0)\n"
              << "  --no-gui             do not open a window\n";
}

static bool parseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (s == "--accum-us" && (v = next())) {
            std::stringstream ss(v);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                const long long n = std::atoll(tok.c_str());
                if (n < 1) { std::cerr << "accum must be >= 1 us\n"; return false; }
                a.accum.push_back(n);
            }
        } else if (s == "--speed" && (v = next())) a.speed = std::atof(v);
        else if (s == "--fps" && (v = next())) a.fps = std::max(1, std::atoi(v));
        else if (s == "--scale" && (v = next())) a.scale = std::max(1, std::atoi(v));
        else if (s == "--dump" && (v = next())) a.dumpDir = v;
        else if (s == "--step-us" && (v = next())) a.step = std::atoll(v);
        else if (s == "--max-frames" && (v = next())) a.maxFrames = std::max(1, std::atoi(v));
        else if (s == "--csv" && (v = next())) a.csv = v;
        else if (s == "--light") a.light = true;
        else if (s == "--no-gui") a.noGui = true;
        else if (s == "--no-loop") a.noLoop = true;
        else if (!s.empty() && s[0] != '-' && a.file.empty()) a.file = s;
        else { std::cerr << "Bad argument: " << s << "\n"; return false; }
    }
    return !a.file.empty();
}

// ------------------------------------------------------------------ headless dump
static int dumpFrames(const std::vector<evt3::Event>& ev, int W, int H, const Args& a, const Palette& pal,
                      const std::vector<int64_t>& accums) {
    fs::create_directories(a.dumpDir);
    const int64_t t0 = ev.front().t, t1 = ev.back().t + 1;
    std::vector<uint8_t> img;
    for (const int64_t acc : accums) {
        const int64_t step = a.step > 0 ? a.step : acc;
        int n = 0;
        for (int64_t t = t0 + step; n < a.maxFrames; t += step, ++n) {
            const int64_t tEnd = std::min(t, t1);
            renderFrame(ev, tEnd, acc, W, H, pal, img);
            char name[64];
            std::snprintf(name, sizeof name, "a%lldus_%05d", static_cast<long long>(acc), n);
            saveImage((fs::path(a.dumpDir) / name).string(), img, W, H);
            if (t >= t1) { ++n; break; }
        }
        std::cout << "accum " << fmtTime(static_cast<double>(acc)) << ": " << n << " frames -> " << a.dumpDir << "\n";
    }
    return 0;
}

// ------------------------------------------------------------------ interactive viewer
#ifdef HAVE_OPENCV
static double accumFromTrack(int v) { return std::max(1.0, std::round(std::pow(10.0, v / 100.0))); }  // 1 us .. 10 s
static int trackFromAccum(double us) { return static_cast<int>(std::lround(100.0 * std::log10(std::max(1.0, us)))); }
static double speedFromTrack(int v) { return std::pow(10.0, (v - 400) / 100.0); }                    // 1e-4 .. 1e2
static int trackFromSpeed(double s) { return static_cast<int>(std::lround(100.0 * std::log10(s) + 400.0)); }

static int runGui(const std::vector<evt3::Event>& ev, int W, int H, const Args& a, const Palette& pal,
                  int64_t accum0) {
    const int64_t t0 = ev.front().t, t1 = ev.back().t + 1;
    const double span = static_cast<double>(std::max<int64_t>(1, t1 - t0));

    double speed = a.speed > 0 ? a.speed : std::min(1.0, span / 1e7);  // default: whole file in ~10 s
    speed = std::min(100.0, std::max(1e-4, speed));
    int scale = a.scale > 0 ? a.scale : std::max(1, std::min(960 / W, 600 / H));

    {
        const char* d = std::getenv("DISPLAY");
        const char* w = std::getenv("WAYLAND_DISPLAY");
        std::cout << "GUI: DISPLAY=" << (d ? d : "(unset)") << " WAYLAND_DISPLAY=" << (w ? w : "(unset)")
                  << ", window " << W * scale << "x" << H * scale << ", opening...\n";
#ifdef __linux__
        if (!d && !w)
            std::cerr << "Warning: no display server is set, so no window can appear (SSH without -X, serial "
                         "console, or a headless board). Use --dump DIR to write frames instead.\n";
#endif
    }
    const std::string win = "EVT3 viewer";
    const char *tAcc = "accum 1us-10s (log)", *tSpd = "speed 1e-4x-100x (log)", *tPos = "position";
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar(tAcc, win, nullptr, 700);
    cv::createTrackbar(tSpd, win, nullptr, 600);
    cv::createTrackbar(tPos, win, nullptr, 1000);
    cv::setTrackbarPos(tAcc, win, std::min(700, trackFromAccum(static_cast<double>(accum0))));
    cv::setTrackbarPos(tSpd, win, std::min(600, std::max(0, trackFromSpeed(speed))));
    cv::setTrackbarPos(tPos, win, 0);

    std::cout << "Keys: space pause | + / - accum x2 | ] / [ speed | a / d step | r restart | l loop | p snapshot | q quit\n";

    double now = static_cast<double>(t0);
    bool paused = false, loop = !a.noLoop;
    int lastPos = 0, snap = 0;
    bool seenVisible = false, firstFrame = true;
    const char* why = "loop ended";
    std::vector<uint8_t> img;
    auto last = std::chrono::steady_clock::now();

    for (;;) {
        const auto wall = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(wall - last).count();
        last = wall;

        const double accum = accumFromTrack(cv::getTrackbarPos(tAcc, win));
        speed = speedFromTrack(cv::getTrackbarPos(tSpd, win));
        const int pos = cv::getTrackbarPos(tPos, win);

        if (pos != lastPos) {  // user dragged the position slider
            now = static_cast<double>(t0) + span * pos / 1000.0;
        } else if (!paused) {
            now += dt * speed * 1e6;
            if (now >= static_cast<double>(t1)) {
                if (loop) now = static_cast<double>(t0);
                else { now = static_cast<double>(t1); paused = true; }
            }
        }

        const size_t n = renderFrame(ev, static_cast<int64_t>(now), static_cast<int64_t>(accum), W, H, pal, img);

        cv::Mat view(H, W, CV_8UC3, img.data());
        cv::Mat shown;
        if (scale > 1) cv::resize(view, shown, cv::Size(W * scale, H * scale), 0, 0, cv::INTER_NEAREST);
        else shown = view.clone();
        char txt[200];
        std::snprintf(txt, sizeof txt, "t=%s / %s   accum=%s   speed=%.4gx   events=%zu%s",
                      fmtTime(now - static_cast<double>(t0)).c_str(), fmtTime(span).c_str(), fmtTime(accum).c_str(),
                      speed, n, paused ? "   [paused]" : "");
        const cv::Scalar fg = a.light ? cv::Scalar(40, 40, 40) : cv::Scalar(220, 220, 220);
        cv::putText(shown, txt, cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, fg, 1);
        cv::imshow(win, shown);
        if (firstFrame) {
            std::cout << "First frame shown (" << n << " events in the window). If no window is visible, check the "
                         "display / X forwarding setup.\n";
            firstFrame = false;
        }

        const int newPos = static_cast<int>(std::lround((now - static_cast<double>(t0)) / span * 1000.0));
        if (newPos != pos) cv::setTrackbarPos(tPos, win, newPos);
        lastPos = cv::getTrackbarPos(tPos, win);

        int k = cv::waitKey(std::max(1, 1000 / a.fps));
        if (k >= 0) k &= 0xFF;
        if (k == 'q' || k == 27) { why = "quit key"; break; }
        // Only treat "not visible" as closed after the window was seen once: some backends report
        // 0 or -1 for this property, which used to end the viewer immediately.
        if (cv::getWindowProperty(win, cv::WND_PROP_VISIBLE) >= 1) seenVisible = true;
        else if (seenVisible) { why = "window closed"; break; }
        switch (k) {
            case ' ': paused = !paused; break;
            case 'l': loop = !loop; break;
            case 'r': now = static_cast<double>(t0); paused = false; break;
            case '+': case '=': cv::setTrackbarPos(tAcc, win, std::min(700, cv::getTrackbarPos(tAcc, win) + 30)); break;
            case '-': case '_': cv::setTrackbarPos(tAcc, win, std::max(0, cv::getTrackbarPos(tAcc, win) - 30)); break;
            case ']': cv::setTrackbarPos(tSpd, win, std::min(600, cv::getTrackbarPos(tSpd, win) + 30)); break;
            case '[': cv::setTrackbarPos(tSpd, win, std::max(0, cv::getTrackbarPos(tSpd, win) - 30)); break;
            case 'a': now = std::max(static_cast<double>(t0), now - accum); break;
            case 'd': now = std::min(static_cast<double>(t1), now + accum); break;
            case 'p': {
                char name[48];
                std::snprintf(name, sizeof name, "evt3_snapshot_%03d", snap++);
                std::cout << "saved " << saveImage(name, img, W, H) << "\n";
                break;
            }
            default: break;
        }
    }
    cv::destroyAllWindows();
    std::cout << "Viewer finished (" << why << ").\n";
    return 0;
}
#endif

// ------------------------------------------------------------------ main
int main(int argc, char** argv) {
    Args a;
    if (!parseArgs(argc, argv, a)) { usage(argv[0]); return 1; }

    evt3::Info info;
    std::vector<evt3::Event> ev;
    std::string err;
    if (!evt3::load(a.file, info, ev, err)) { std::cerr << "Error: " << err << "\n"; return 1; }
    if (ev.empty()) { std::cerr << "No events in " << a.file << "\n"; return 1; }

    size_t on = 0;
    for (const auto& e : ev) on += e.p;
    const int64_t t0 = ev.front().t, t1 = ev.back().t + 1;
    std::cout << a.file << ": " << ev.size() << " events (" << on << " ON / " << ev.size() - on << " OFF), "
              << info.width << "x" << info.height << ", duration " << fmtTime(static_cast<double>(t1 - t0))
              << ", t0=" << t0 << " us";
    if (info.ignored) std::cout << ", " << info.ignored << " non-CD words skipped";
    std::cout << "\n";

    if (!a.csv.empty()) {
        std::ofstream f(a.csv);
        f << "t,x,y,polarity\n";
        for (const auto& e : ev) f << e.t << "," << e.x << "," << e.y << "," << static_cast<int>(e.p) << "\n";
        std::cout << "wrote " << a.csv << " (time in us)\n";
    }

    const Palette& pal = a.light ? kLight : kDark;
    std::vector<int64_t> accums = a.accum;
    if (accums.empty())  // auto: 10 ms, but never more than a tenth of the recording
        accums.push_back(std::max<int64_t>(1, std::min<int64_t>(10000, (t1 - t0) / 10)));

    if (!a.dumpDir.empty()) return dumpFrames(ev, info.width, info.height, a, pal, accums);
    if (a.noGui) return 0;
#ifdef HAVE_OPENCV
    try {
        return runGui(ev, info.width, info.height, a, pal, accums.front());
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what()
                  << "\nThis usually means OpenCV was built without GUI support (no GTK/Qt) or there is no display. "
                     "Use --dump DIR instead.\n";
        return 1;
    }
#else
    std::cerr << "This build has no OpenCV, so there is no interactive window. "
                 "Rebuild with OpenCV installed, or use --dump DIR / --csv FILE --no-gui.\n";
    return 1;
#endif
}
