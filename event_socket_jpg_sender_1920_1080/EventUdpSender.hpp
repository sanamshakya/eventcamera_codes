/*
 * EventUdpSender - accumulates evsim::Event records into an image (same
 * red=ON/blue=OFF convention as EventVisualizer and
 * visualize_events_multi_accum.py), JPEG-encodes it, and sends it as a
 * single UDP datagram to a host PC for display there.
 *
 * Deliberately uses ONLY opencv core + imgproc + imgcodecs - no highgui.
 * highgui is what pulls in a GTK (or Qt) runtime dependency; imencode()
 * needs none of that. This is the fix for "opencv visualization fails,
 * GTK library unavailable" on the target: don't link highgui there at
 * all, encode + stream instead of displaying locally.
 *
 * Opt-in via EVSIM_ENABLE_CV_UDP (separate from EVSIM_ENABLE_CV_VIS, which
 * guards the highgui-based EventVisualizer - keep that one OFF on this
 * target). Build with:
 *
 *   -DEVSIM_ENABLE_CV_UDP $(pkg-config --cflags opencv4)
 *   -lopencv_core -lopencv_imgproc -lopencv_imgcodecs   # NOT opencv_highgui
 *
 * Companion host-side viewer: receive_events_udp.py
 */

#ifndef EVENTUDPSENDER_HPP
#define EVENTUDPSENDER_HPP

#ifdef EVSIM_ENABLE_CV_UDP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <iostream>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "EventPacket.h"

class EventUdpSender
{
public:
    // hostIp/port: where accumulated JPEG frames are sent (run
    // receive_events_udp.py there). accumWindowMs: same event-timestamp
    // accumulation window as EventVisualizer. jpegQuality: 0-100.
    bool Init(int width, int height, const std::string &hostIp, int port,
              double accumWindowMs = 20.0, int jpegQuality = 80)
    {
        m_width          = width;
        m_height         = height;
        m_accumWindowSec = accumWindowMs / 1000.0;
        m_jpegQuality    = jpegQuality;

        m_accum = cv::Mat::zeros(height, width, CV_32FC3);
        m_windowStart = -1.0;

        m_sockFd = socket(AF_INET, SOCK_DGRAM, 0);
        if (m_sockFd < 0)
        {
            std::cerr << "EventUdpSender: socket() failed\n";
            return false;
        }

        std::memset(&m_destAddr, 0, sizeof(m_destAddr));
        m_destAddr.sin_family = AF_INET;
        m_destAddr.sin_port   = htons(static_cast<uint16_t>(port));

        if (inet_pton(AF_INET, hostIp.c_str(), &m_destAddr.sin_addr) != 1)
        {
            std::cerr << "EventUdpSender: invalid host IP '" << hostIp << "'\n";
            close(m_sockFd);
            m_sockFd = -1;
            return false;
        }

        m_initialized = true;
        std::cout << "EventUdpSender: streaming to " << hostIp << ":" << port << "\n";
        return true;
    }

    // Feed one packet's events in. Same time-windowed accumulate/reset
    // logic as EventVisualizer::Feed(), but sends over UDP instead of
    // calling imshow.
    void Feed(const evsim::EventPacket &packet)
    {
        if (!m_initialized)
            return;

        for (const auto &ev : packet.events)
        {
            if (m_windowStart < 0.0)
                m_windowStart = ev.timestamp;

            if (ev.x < m_width && ev.y < m_height)
            {
                cv::Vec3f &px = m_accum.at<cv::Vec3f>(ev.y, ev.x);
                if (ev.polarity > 0)
                    px[2] += 1.0f;
                else
                    px[0] += 1.0f;
            }

            if (ev.timestamp - m_windowStart >= m_accumWindowSec)
                RenderAndSend(ev.timestamp);
        }

        if (m_windowStart >= 0.0 && packet.endTime - m_windowStart >= m_accumWindowSec)
            RenderAndSend(packet.endTime);
    }

    void Deinit()
    {
        if (m_sockFd >= 0)
        {
            close(m_sockFd);
            m_sockFd = -1;
        }
        m_initialized = false;
        std::cout << "EventUdpSender: sent " << m_uFramesSent << " frame(s), dropped "
                  << m_uFramesDroppedTooLarge << " (too large) / "
                  << m_uFramesDroppedSendFail << " (send failed)\n";
    }

    uint64_t GetFramesSent() const { return m_uFramesSent; }

private:
    // Safe under the 65507-byte UDP payload ceiling (65535 - 8-byte UDP
    // header - 20-byte IPv4 header), with margin. One accumulation frame
    // is sent as exactly one datagram - no chunking/reassembly - so an
    // oversized frame is simply dropped rather than split. Mostly-black
    // sparse-event images compress well, so this should be rare in
    // practice; if you see frequent drops, lower jpegQuality or the
    // accumulation window (fewer events per frame -> smaller JPEG).
    static constexpr size_t kMaxUdpPayload = 60000;

    void RenderAndSend(double windowEndTimestamp)
    {
        double maxVal = 0.0;
        cv::minMaxLoc(m_accum.reshape(1), nullptr, &maxVal);

        cv::Mat display;
        if (maxVal > 0.0)
            m_accum.convertTo(display, CV_8UC3, 255.0 / maxVal);
        else
            m_accum.convertTo(display, CV_8UC3);

        std::vector<uint8_t> jpegBuf;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, m_jpegQuality};
        cv::imencode(".jpg", display, jpegBuf, params);

        m_accum.setTo(cv::Scalar(0, 0, 0));
        m_windowStart = windowEndTimestamp;

        if (jpegBuf.size() > kMaxUdpPayload)
        {
            ++m_uFramesDroppedTooLarge;
            std::cerr << "EventUdpSender: encoded frame " << jpegBuf.size()
                      << " bytes exceeds " << kMaxUdpPayload << " byte limit, dropping\n";
            return;
        }

        const ssize_t sent = sendto(m_sockFd, jpegBuf.data(), jpegBuf.size(), 0,
                                     reinterpret_cast<sockaddr *>(&m_destAddr),
                                     sizeof(m_destAddr));
        if (sent < 0)
        {
            ++m_uFramesDroppedSendFail;
            std::cerr << "EventUdpSender: sendto() failed\n";
            return;
        }

        ++m_uFramesSent;
    }

    int    m_width  = 0;
    int    m_height = 0;
    double m_accumWindowSec = 0.02;
    double m_windowStart = -1.0;
    int    m_jpegQuality = 80;
    cv::Mat m_accum;
    bool   m_initialized = false;

    int         m_sockFd = -1;
    sockaddr_in m_destAddr{};

    uint64_t m_uFramesSent = 0;
    uint64_t m_uFramesDroppedTooLarge = 0;
    uint64_t m_uFramesDroppedSendFail = 0;
};

#endif // EVSIM_ENABLE_CV_UDP
#endif // EVENTUDPSENDER_HPP
