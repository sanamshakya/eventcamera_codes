/*
 * EventVisualizer - accumulates evsim::Event records into an OpenCV image
 * (red=ON, blue=OFF, matching the offline visualize_events_multi_accum.py
 * convention) and displays it, refreshing every `accumWindowMs` of EVENT
 * TIMESTAMP (not wall-clock) rather than every packet.
 *
 * Opt-in: this whole file compiles to nothing unless EVSIM_ENABLE_CV_VIS is
 * defined, so the default build keeps the no-OpenCV footprint intact (see
 * the MinMaxU16 comment in CNvSIPLConsumer.hpp - OpenCV was deliberately
 * kept out of this codebase). Build with:
 *
 *   -DEVSIM_ENABLE_CV_VIS $(pkg-config --cflags opencv4)
 *   $(pkg-config --libs opencv4)   # needs core, imgproc, highgui
 *
 * On a headless embedded target (no X11/Wayland), cv::imshow will fail to
 * open a window - this is meant for interactive dev/debug sessions, not
 * the production headless capture path.
 */

#ifndef EVENTVISUALIZER_HPP
#define EVENTVISUALIZER_HPP

#ifdef EVSIM_ENABLE_CV_VIS

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

#include "EventPacket.h"

class EventVisualizer
{
public:
    // accumWindowMs: span of EVENT timestamps to accumulate before each
    // render+clear. Independent of how often generate() is actually
    // called - a single packet spanning multiple windows renders multiple
    // times inside Feed(); a packet narrower than one window just keeps
    // accumulating until a later packet crosses the threshold.
    bool Init(int width, int height, double accumWindowMs,
              const std::string &windowName = "Events")
    {
        m_width          = width;
        m_height         = height;
        m_accumWindowSec = accumWindowMs / 1000.0;
        m_windowName     = windowName;

        m_accum = cv::Mat::zeros(height, width, CV_32FC3);
        m_windowStart = -1.0;

        cv::namedWindow(m_windowName, cv::WINDOW_NORMAL);

        m_initialized = true;
        return true;
    }

    // Feed one packet's events in. Returns false if the display window was
    // closed / 'q' or ESC was pressed - caller should stop calling Feed()
    // (and probably tear down visualization) once that happens.
    bool Feed(const evsim::EventPacket &packet)
    {
        if (!m_initialized)
            return true;

        for (const auto &ev : packet.events)
        {
            if (m_windowStart < 0.0)
                m_windowStart = ev.timestamp;

            if (ev.x < m_width && ev.y < m_height)
            {
                // OpenCV Mats are BGR - channel 2 = red (ON), channel 0 = blue (OFF).
                cv::Vec3f &px = m_accum.at<cv::Vec3f>(ev.y, ev.x);
                if (ev.polarity > 0)
                    px[2] += 1.0f;
                else
                    px[0] += 1.0f;
            }

            if (ev.timestamp - m_windowStart >= m_accumWindowSec)
            {
                if (!RenderAndReset(ev.timestamp))
                    return false;
            }
        }

        // A packet with few/no events won't cross the window on its own -
        // check against packet.endTime too so a quiet scene still refreshes
        // roughly on schedule instead of holding a stale accumulation.
        if (m_windowStart >= 0.0 && packet.endTime - m_windowStart >= m_accumWindowSec)
        {
            if (!RenderAndReset(packet.endTime))
                return false;
        }

        return true;
    }

    void Deinit()
    {
        if (m_initialized)
        {
            cv::destroyWindow(m_windowName);
            m_initialized = false;
        }
    }

private:
    bool RenderAndReset(double windowEndTimestamp)
    {
        double maxVal = 0.0;
        cv::minMaxLoc(m_accum.reshape(1), nullptr, &maxVal); // reshape: min/max across all 3 channels together

        cv::Mat display;
        if (maxVal > 0.0)
            m_accum.convertTo(display, CV_8UC3, 255.0 / maxVal);
        else
            m_accum.convertTo(display, CV_8UC3);

        cv::imshow(m_windowName, display);
        const int key = cv::waitKey(1);

        m_accum.setTo(cv::Scalar(0, 0, 0));
        m_windowStart = windowEndTimestamp;

        if (key == 'q' || key == 27) // ESC
            return false;
        return true;
    }

    int         m_width  = 0;
    int         m_height = 0;
    double      m_accumWindowSec = 0.02;
    double      m_windowStart = -1.0;
    std::string m_windowName;
    cv::Mat     m_accum;
    bool        m_initialized = false;
};

#endif // EVSIM_ENABLE_CV_VIS
#endif // EVENTVISUALIZER_HPP
