/*
 * EventRawUdpSender - accumulates evsim::Event records into a plain 8-bit
 * single-channel buffer (no OpenCV Mat, no JPEG encode - just a
 * std::vector<uint8_t>) and streams it to a host PC over UDP, split into
 * fixed-size chunks with a small header since one frame (e.g. 1280x720 =
 * ~920KB) is far larger than a single UDP datagram can carry.
 *
 * Pixel encoding: 1 byte/pixel, centered at 128 ("no activity in this
 * window"). Brighter (>128) = net ON events at that pixel, darker (<128)
 * = net OFF events, normalized per-frame by the largest |ON-OFF| count so
 * contrast stays consistent frame to frame. This is a third the bandwidth
 * of a 3-channel BGR buffer for the same resolution.
 *
 * No OpenCV needed at all on this side - opt-in via EVSIM_ENABLE_RAW_UDP
 * purely so it's consistent with the other optional visualization paths
 * and easy to compile out entirely. Companion host-side viewer:
 * receive_events_udp.py (raw-chunk-reassembly version).
 */

#ifndef EVENTRAWUDPSENDER_HPP
#define EVENTRAWUDPSENDER_HPP

#ifdef EVSIM_ENABLE_RAW_UDP

#include <vector>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <iostream>
#include <string>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "EventPacket.h"

#pragma pack(push, 1)
struct RawFrameChunkHeader
{
    uint32_t magic = 0x45465248u; // 'EFRH'
    uint32_t frameSeq;
    uint32_t width;
    uint32_t height;
    uint32_t totalChunks;
    uint32_t chunkIndex;
    uint32_t chunkBytes; // payload bytes that follow this header in this datagram
};
#pragma pack(pop)

class EventRawUdpSender
{
public:
    // hostIp/port: where raw frames are streamed (run receive_events_udp.py
    // there). accumWindowMs: event-timestamp accumulation window, same
    // convention as the other senders. chunkPayloadBytes: bytes of frame
    // data per UDP datagram (default 60000, safely under the ~65507-byte
    // UDP payload ceiling). Smaller chunks avoid IP-level fragmentation
    // (~1400 bytes stays under a standard 1500-byte Ethernet MTU) at the
    // cost of many more sendto() calls per frame; larger chunks mean
    // fewer syscalls but losing one IP fragment loses the whole chunk.
    bool Init(int width, int height, const std::string &hostIp, int port,
              double accumWindowMs = 20.0, size_t chunkPayloadBytes = 60000)
    {
        m_width  = width;
        m_height = height;
        m_accumWindowSec = accumWindowMs / 1000.0;
        m_chunkPayloadBytes = chunkPayloadBytes;

        const size_t numPixels = static_cast<size_t>(width) * height;
        m_onCount.assign(numPixels, 0);
        m_offCount.assign(numPixels, 0);
        m_frameBuf.resize(numPixels);

        m_windowStart = -1.0;
        m_frameSeq = 0;

        m_sockFd = socket(AF_INET, SOCK_DGRAM, 0);
        if (m_sockFd < 0)
        {
            std::cerr << "EventRawUdpSender: socket() failed\n";
            return false;
        }

        std::memset(&m_destAddr, 0, sizeof(m_destAddr));
        m_destAddr.sin_family = AF_INET;
        m_destAddr.sin_port   = htons(static_cast<uint16_t>(port));

        if (inet_pton(AF_INET, hostIp.c_str(), &m_destAddr.sin_addr) != 1)
        {
            std::cerr << "EventRawUdpSender: invalid host IP '" << hostIp << "'\n";
            close(m_sockFd);
            m_sockFd = -1;
            return false;
        }

        m_initialized = true;
        std::cout << "EventRawUdpSender: streaming raw " << width << "x" << height
                  << " frames (" << numPixels << " B/frame) to "
                  << hostIp << ":" << port << "\n";
        return true;
    }

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
                const size_t idx = static_cast<size_t>(ev.y) * m_width + ev.x;
                if (ev.polarity > 0)
                    ++m_onCount[idx];
                else
                    ++m_offCount[idx];
            }

            if (ev.timestamp - m_windowStart >= m_accumWindowSec)
                RenderAndSend();
        }

        if (m_windowStart >= 0.0 && packet.endTime - m_windowStart >= m_accumWindowSec)
            RenderAndSend();
    }

    void Deinit()
    {
        if (m_sockFd >= 0)
        {
            close(m_sockFd);
            m_sockFd = -1;
        }
        m_initialized = false;
        std::cout << "EventRawUdpSender: sent " << m_uFramesSent << " frame(s), "
                  << m_uChunksSent << " chunk(s) total\n";
    }

private:
    void RenderAndSend()
    {
        const size_t n = m_frameBuf.size();

        // Find the peak |ON-OFF| this window to normalize contrast, same
        // idea as the maxVal normalization in the JPEG-based senders.
        int32_t maxAbsDiff = 0;
        for (size_t i = 0; i < n; ++i)
        {
            const int32_t diff = static_cast<int32_t>(m_onCount[i]) -
                                  static_cast<int32_t>(m_offCount[i]);
            const int32_t absDiff = diff < 0 ? -diff : diff;
            if (absDiff > maxAbsDiff)
                maxAbsDiff = absDiff;
        }

        const float scale = maxAbsDiff > 0 ? (127.0f / static_cast<float>(maxAbsDiff)) : 0.0f;

        for (size_t i = 0; i < n; ++i)
        {
            const int32_t diff = static_cast<int32_t>(m_onCount[i]) -
                                  static_cast<int32_t>(m_offCount[i]);
            int value = 128 + static_cast<int>(diff * scale);
            value = std::min(255, std::max(0, value));
            m_frameBuf[i] = static_cast<uint8_t>(value);
        }

        std::fill(m_onCount.begin(), m_onCount.end(), 0);
        std::fill(m_offCount.begin(), m_offCount.end(), 0);
        m_windowStart = -1.0; // set again on the next event fed in

        SendChunked();
        ++m_frameSeq;
    }

    void SendChunked()
    {
        const size_t totalBytes = m_frameBuf.size();
        const uint32_t totalChunks = static_cast<uint32_t>(
            (totalBytes + m_chunkPayloadBytes - 1) / m_chunkPayloadBytes);

        std::vector<uint8_t> packetBuf(sizeof(RawFrameChunkHeader) + m_chunkPayloadBytes);

        for (uint32_t chunkIdx = 0; chunkIdx < totalChunks; ++chunkIdx)
        {
            const size_t offset = static_cast<size_t>(chunkIdx) * m_chunkPayloadBytes;
            const size_t thisChunkBytes = std::min(m_chunkPayloadBytes, totalBytes - offset);

            RawFrameChunkHeader hdr;
            hdr.frameSeq     = m_frameSeq;
            hdr.width        = static_cast<uint32_t>(m_width);
            hdr.height       = static_cast<uint32_t>(m_height);
            hdr.totalChunks  = totalChunks;
            hdr.chunkIndex   = chunkIdx;
            hdr.chunkBytes   = static_cast<uint32_t>(thisChunkBytes);

            std::memcpy(packetBuf.data(), &hdr, sizeof(hdr));
            std::memcpy(packetBuf.data() + sizeof(hdr), m_frameBuf.data() + offset, thisChunkBytes);

            const ssize_t sent = sendto(m_sockFd, packetBuf.data(), sizeof(hdr) + thisChunkBytes, 0,
                                         reinterpret_cast<sockaddr *>(&m_destAddr), sizeof(m_destAddr));
            if (sent < 0)
            {
                std::cerr << "EventRawUdpSender: sendto() failed on chunk "
                          << chunkIdx << "/" << totalChunks << "\n";
                continue; // best-effort - the frame will just be incomplete on the receiver
            }
            ++m_uChunksSent;
        }

        ++m_uFramesSent;
    }

    int    m_width  = 0;
    int    m_height = 0;
    double m_accumWindowSec = 0.02;
    double m_windowStart = -1.0;
    size_t m_chunkPayloadBytes = 60000;

    std::vector<uint32_t> m_onCount;
    std::vector<uint32_t> m_offCount;
    std::vector<uint8_t>  m_frameBuf;

    uint32_t m_frameSeq = 0;

    int         m_sockFd = -1;
    sockaddr_in m_destAddr{};

    bool     m_initialized = false;
    uint64_t m_uFramesSent = 0;
    uint64_t m_uChunksSent = 0;
};

#endif // EVSIM_ENABLE_RAW_UDP
#endif // EVENTRAWUDPSENDER_HPP
