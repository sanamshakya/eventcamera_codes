/*
 * CEventFileWriter - writes evsim::EventPacket streams to a simple
 * fixed-record binary file for offline verification/visualization.
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
    // width/height must be known (i.e. call this after EventGenerator::initialize()
    // has run at least once) so the header can be written correctly.
    bool Init(const string &sFilename, uint32_t width, uint32_t height)
    {
        remove(sFilename.c_str());
        m_pOutFile = fopen(sFilename.c_str(), "wb");
        if (!m_pOutFile) {
            cerr << "CEventFileWriter: failed to create " << sFilename << endl;
            return false;
        }

        EventFileHeader header;
        header.width  = width;
        header.height = height;

        if (fwrite(&header, sizeof(header), 1U, m_pOutFile) != 1U) {
            cerr << "CEventFileWriter: failed to write header\n";
            return false;
        }

        m_initialized = true;
        return true;
    }

    bool WriteEventPacket(const evsim::EventPacket &packet)
    {
        if (!m_initialized || m_pOutFile == nullptr) {
            cerr << "CEventFileWriter: WriteEventPacket called before Init\n";
            return false;
        }

        if (packet.events.empty()) {
            return true; // nothing to write, not an error
        }

        // NOTE: ASSUMPTION - evsim event struct fields are named x, y,
        // timestamp, polarity, matching the push_back(...) call site in
        // EventGenerator.cpp: {x, y, eventTime, +1/-1}. If EventPacket.h
        // uses different field names, adjust the three lines below.
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
        return true;
    }

    uint64_t GetTotalEventsWritten() const
    {
        return m_uTotalEventsWritten;
    }

    void Deinit(void)
    {
        if (m_pOutFile != nullptr) {
            fflush(m_pOutFile);
            fclose(m_pOutFile);
            m_pOutFile = nullptr;
        }
        cout << "CEventFileWriter: wrote " << m_uTotalEventsWritten << " events total\n";
    }

private:
    FILE     *m_pOutFile = nullptr;
    bool      m_initialized = false;
    uint64_t  m_uTotalEventsWritten = 0u;
};

#endif //CEVENTFILEWRITER_HPP
