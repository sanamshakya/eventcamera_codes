/*
 * CEventFileWriter - writes evsim::EventPacket streams to a simple
 * fixed-record binary file for offline verification/visualization.
 *
 * Splits output across multiple files instead of one growing file: a new
 * file is opened every `framesPerFile` EventPacket frames, named from the
 * frame number the file starts at, so filenames stay directly traceable to
 * the "frame %llu produced ..." log lines elsewhere.
 */

#include <cstdio>
#include <cstdint>
#include <string>
#include <iostream>

#include "EventPacket.h"

#ifndef CEVENTFILEWRITER_HPP
#define CEVENTFILEWRITER_HPP

using namespace std;

#pragma pack(push, 1)
struct EventFileHeader
{
    uint32_t magic   = 0x45565342u; // 'EVSB'
    uint32_t version = 1u;
    uint32_t width   = 0u;
    uint32_t height  = 0u;
};

struct EventFileRecord
{
    uint16_t x;
    uint16_t y;
    double   timestamp;
    int8_t   polarity;
};
#pragma pack(pop)

class CEventFileWriter
{
public:
    // sBaseFilename is used as a prefix - each output file is named
    // "<sBaseFilename>_frame<N>.<ext>" where N is the frame number the
    // file starts at (if sBaseFilename has an extension, e.g. "events.evsb",
    // it's preserved: "events_frame000123.evsb"; otherwise ".evsb" is
    // appended).
    //
    // framesPerFile controls the split granularity:
    //   1        -> one file per frame (default - what you asked for)
    //   N > 1     -> batch N frames per file (fewer, larger files)
    //   0        -> old behaviour: a single file for the whole run
    bool Init(const string &sBaseFilename, uint32_t width, uint32_t height,
              uint32_t framesPerFile = 1)
    {
        m_sBaseFilename  = sBaseFilename;
        m_uWidth         = width;
        m_uHeight        = height;
        m_uFramesPerFile = framesPerFile; // 0 == never rotate

        SplitBaseAndExtension();

        m_initialized = true;
        return true; // first file is opened lazily, on the first WriteEventPacket
    }

    bool WriteEventPacket(const evsim::EventPacket &packet)
    {
        if (!m_initialized) {
            cerr << "CEventFileWriter: WriteEventPacket called before Init\n";
            return false;
        }

        const bool needsRotate =
            (m_pOutFile == nullptr) ||
            (m_uFramesPerFile != 0 && m_uFramesInCurrentFile >= m_uFramesPerFile);

        if (needsRotate) {
            if (!RotateFile(packet.frameNumber)) {
                return false;
            }
        }

        if (!packet.events.empty()) {
            for (const auto &ev : packet.events) {
                EventFileRecord rec;
                rec.x         = ev.x;
                rec.y         = ev.y;
                rec.timestamp = ev.timestamp;
                rec.polarity  = static_cast<int8_t>(ev.polarity);

                if (fwrite(&rec, sizeof(rec), 1U, m_pOutFile) != 1U) {
                    cerr << "CEventFileWriter: event write failed\n";
                    return false;
                }
            }

            m_uTotalEventsWritten += packet.events.size();
        }

        ++m_uFramesInCurrentFile;
        ++m_uTotalFramesWritten;
        return true;
    }

    uint64_t GetTotalEventsWritten() const
    {
        return m_uTotalEventsWritten;
    }

    uint64_t GetTotalFramesWritten() const
    {
        return m_uTotalFramesWritten;
    }

    void Deinit(void)
    {
        CloseCurrentFile();
        cout << "CEventFileWriter: wrote " << m_uTotalEventsWritten
             << " events across " << m_uTotalFramesWritten << " frames in "
             << m_uFilesWritten << " file(s)\n";
    }

private:
    void SplitBaseAndExtension()
    {
        const size_t slash = m_sBaseFilename.find_last_of("/\\");
        const size_t dot   = m_sBaseFilename.find_last_of('.');

        // Only treat it as an extension if the dot comes after the last
        // path separator (so "/data/run.1/events" isn't split on that dot).
        if (dot != string::npos && (slash == string::npos || dot > slash)) {
            m_sBaseNoExt = m_sBaseFilename.substr(0, dot);
            m_sExt       = m_sBaseFilename.substr(dot); // includes the '.'
        } else {
            m_sBaseNoExt = m_sBaseFilename;
            m_sExt       = ".evsb";
        }
    }

    bool RotateFile(uint64_t startFrameNumber)
    {
        CloseCurrentFile();

        char suffix[64];
        snprintf(suffix, sizeof(suffix), "_frame%06llu",
                 (unsigned long long)startFrameNumber);

        const string sFilename = m_sBaseNoExt + suffix + m_sExt;

        remove(sFilename.c_str());
        m_pOutFile = fopen(sFilename.c_str(), "wb");
        if (!m_pOutFile) {
            cerr << "CEventFileWriter: failed to create " << sFilename << endl;
            return false;
        }

        EventFileHeader header;
        header.width  = m_uWidth;
        header.height = m_uHeight;

        if (fwrite(&header, sizeof(header), 1U, m_pOutFile) != 1U) {
            cerr << "CEventFileWriter: failed to write header for " << sFilename << endl;
            fclose(m_pOutFile);
            m_pOutFile = nullptr;
            return false;
        }

        m_uFramesInCurrentFile = 0;
        ++m_uFilesWritten;

        cout << "CEventFileWriter: opened " << sFilename << endl;
        return true;
    }

    void CloseCurrentFile()
    {
        if (m_pOutFile != nullptr) {
            fflush(m_pOutFile);
            fclose(m_pOutFile);
            m_pOutFile = nullptr;
        }
    }

    FILE     *m_pOutFile = nullptr;
    bool      m_initialized = false;

    string    m_sBaseFilename;
    string    m_sBaseNoExt;
    string    m_sExt;

    uint32_t  m_uWidth  = 0u;
    uint32_t  m_uHeight = 0u;
    uint32_t  m_uFramesPerFile = 1u;
    uint32_t  m_uFramesInCurrentFile = 0u;

    uint64_t  m_uTotalEventsWritten = 0u;
    uint64_t  m_uTotalFramesWritten = 0u;
    uint64_t  m_uFilesWritten = 0u;
};

#endif //CEVENTFILEWRITER_HPP
