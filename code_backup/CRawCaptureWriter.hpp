/*
 * CRawCaptureWriter - dumps raw Bayer16 frames (as extracted by
 * CNvSIPLConsumer::CaptureRawBayerFrame) to a binary file for offline
 * analysis with visualize_raw_bayer.py, before wiring raw data into
 * EventGenerator.
 *
 * Format:
 *   Header (16 bytes): magic(u32) 'RAWB', version(u32)=1, width(u32), height(u32)
 *   Then N frames, each exactly width*height*2 bytes (raw uint16 pixels,
 *   row-major, no padding between frames).
 */

#include <cstdio>
#include <cstdint>
#include <string>
#include <iostream>
#include <vector>

#ifndef CRAWCAPTUREWRITER_HPP
#define CRAWCAPTUREWRITER_HPP

using namespace std;

#pragma pack(push, 1)
struct RawCaptureHeader
{
    uint32_t magic   = 0x42574152u; // 'RAWB'
    uint32_t version = 1u;
    uint32_t width   = 0u;
    uint32_t height  = 0u;
};
#pragma pack(pop)

class CRawCaptureWriter
{
public:
    // maxFrames: stop accepting frames after this many (0 = unlimited).
    // Keep this small (e.g. 5-10) - raw16 frames are large and this is
    // only for offline analysis, not a full capture pipeline.
    bool Init(const string &sFilename, uint32_t width, uint32_t height, uint32_t maxFrames = 10)
    {
        remove(sFilename.c_str());
        m_pOutFile = fopen(sFilename.c_str(), "wb");
        if (!m_pOutFile) {
            cerr << "CRawCaptureWriter: failed to create " << sFilename << endl;
            return false;
        }

        RawCaptureHeader header;
        header.width  = width;
        header.height = height;

        if (fwrite(&header, sizeof(header), 1U, m_pOutFile) != 1U) {
            cerr << "CRawCaptureWriter: failed to write header\n";
            return false;
        }

        m_width = width;
        m_height = height;
        m_maxFrames = maxFrames;
        m_initialized = true;
        return true;
    }

    // Returns false once maxFrames reached (caller should stop calling /
    // close the writer) - not itself an error.
    bool WriteFrame(const std::vector<uint16_t> &raw16, uint32_t width, uint32_t height)
    {
        if (!m_initialized || m_pOutFile == nullptr) {
            cerr << "CRawCaptureWriter: WriteFrame called before Init\n";
            return false;
        }
        if ((m_maxFrames != 0) && (m_uFramesWritten >= m_maxFrames)) {
            return false;
        }
        if ((width != m_width) || (height != m_height) ||
            (raw16.size() != static_cast<size_t>(width) * height)) {
            cerr << "CRawCaptureWriter: frame size mismatch "
                 << "(got " << width << "x" << height
                 << ", expected " << m_width << "x" << m_height << ")\n";
            return false;
        }

        size_t bytesExpected = static_cast<size_t>(m_width) * m_height * sizeof(uint16_t);
        if (fwrite(raw16.data(), 1U, bytesExpected, m_pOutFile) != bytesExpected) {
            cerr << "CRawCaptureWriter: frame write failed\n";
            return false;
        }

        m_uFramesWritten++;
        return true;
    }

    uint32_t GetFramesWritten() const { return m_uFramesWritten; }

    bool IsDone() const
    {
        return (m_maxFrames != 0) && (m_uFramesWritten >= m_maxFrames);
    }

    void Deinit(void)
    {
        if (m_pOutFile != nullptr) {
            fflush(m_pOutFile);
            fclose(m_pOutFile);
            m_pOutFile = nullptr;
        }
        cout << "CRawCaptureWriter: wrote " << m_uFramesWritten << " raw frames\n";
    }

private:
    FILE     *m_pOutFile = nullptr;
    bool      m_initialized = false;
    uint32_t  m_width = 0u;
    uint32_t  m_height = 0u;
    uint32_t  m_maxFrames = 10u;
    uint32_t  m_uFramesWritten = 0u;
};

#endif //CRAWCAPTUREWRITER_HPP
