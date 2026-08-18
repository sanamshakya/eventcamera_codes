/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/* STL Headers */
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

#include "NvSIPLClient.hpp"
#include "CProfiler.hpp"
#include "CFrameFeeder.hpp"
#include "CFileWriter.hpp"

/* Event generation (evsim) */
#include "Config.h"
#include "EventGenerator.h"
#include "EventPacket.h"
#include "CEventFileWriter.hpp"
#include "CRawCaptureWriter.hpp"

#if !NV_IS_SAFETY
#include "CComposite.hpp"
#include "CNvSIPLMasterNvSci.hpp"
#endif // !NV_IS_SAFETY

#ifndef CNVSIPLCONSUMER_HPP
#define CNVSIPLCONSUMER_HPP

#define IMAGE_QUEUE_TIMEOUT_US (1000000U)

using namespace std;
using namespace nvsipl;

/** NvSIPL consumer class.
 * NvSIPL consumer consumes the output buffers of NvSIPL.
 */
class CNvSIPLConsumer
{
public:
    SIPLStatus Init(
#if !NV_IS_SAFETY
        CComposite *pComposite = nullptr,
        CNvSIPLMasterNvSci *pMasterNvSci = nullptr,
#endif // !NV_IS_SAFETY
        uint32_t uID = -1,
        uint32_t uSensor = -1,
        INvSIPLClient::ConsumerDesc::OutputType outputType = INvSIPLClient::ConsumerDesc::OutputType::ICP,
        CProfiler *pProfiler = nullptr,
        CFrameFeeder *pFrameFeeder = nullptr,
        std::string sFilenamePrefix = "",
        uint32_t uNumSkipFrames = 0u,
        uint64_t uNumWriteFrames = -1u)
    {
#if !NV_IS_SAFETY
        if ((pComposite != nullptr) && (pMasterNvSci != nullptr))
        {
            LOG_ERR("CNvSIPLConsumer expects only one of pComposite and pMasterNvSci\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        m_pComposite = pComposite;
        m_pMasterNvSci = pMasterNvSci;
#endif // !NV_IS_SAFETY

        m_uID = uID;
        m_uSensor = uSensor;
        m_outputType = outputType;
        m_pProfiler = pProfiler;
        m_pFrameFeeder = pFrameFeeder;
        m_sFilenamePrefix = sFilenamePrefix;
        m_uNumSkipFrames = uNumSkipFrames;
        m_uNumWriteFrames = uNumWriteFrames;

        // Create file writer if necessary
        if (m_sFilenamePrefix != "")
        {
            m_pFileWriter.reset(new CFileWriter);
        }

        return NVSIPL_STATUS_OK;
    }

    void Deinit(void)
    {
        if (m_pFileWriter != nullptr)
        {
            m_pFileWriter->Deinit();
            m_pFileWriter = nullptr;
        }

        // --- Event generation cleanup (step 1-4) ---
        // (capture buffers are std::vector members - no manual free needed)
        if (m_pEventFileWriter != nullptr)
        {
            m_pEventFileWriter->Deinit();
            m_pEventFileWriter = nullptr;
        }

        // --- Raw capture cleanup ---
        if (m_pRawCaptureWriter != nullptr)
        {
            m_pRawCaptureWriter->Deinit();
            m_pRawCaptureWriter = nullptr;
        }

        return;
    }

    void EnableMetadataLogging(void)
    {
        m_bShowMetadata = true;
    }

    // --- Raw Bayer capture-for-analysis (standalone, independent of
    // event generation) ---
    // Dumps the first `numFrames` raw16 Bayer frames to sFilename, then
    // stops automatically. Use this FIRST, in isolation, to verify the
    // raw capture is correct via visualize_raw_bayer.py before wiring
    // raw data into EventGenerator.
    void EnableRawCapture(const string &sFilename, uint32_t numFrames = 100)
    {
        m_sRawCaptureFilename = sFilename;
        m_uRawCaptureNumFrames = numFrames;
        m_bRawCaptureEnabled = true;
    }

    // --- Event generation setup (step 1 & 4) ---
    // sEventFilename: pass "" to skip binary event-file output (e.g. once
    // you move to the queue-based downstream consumer in the next step).
    void EnableEventGeneration(const evsim::Config &evConfig,
                               const string &sEventFilename = "")
    {
        m_pEventGenerator.reset(new evsim::EventGenerator(evConfig));
        m_sEventFilename = sEventFilename;
        m_bEventGenEnabled = true;
    }

    // TSC ticks-per-second for this platform (step 3).
    // ASSUMPTION: default is a placeholder - verify against your platform's
    // actual oscillator frequency (check CProfiler or other TSC-consuming
    // code in this codebase for the authoritative value) before trusting
    // event timestamps.
    void SetTscFrequency(double freqHz)
    {
        m_dTscFreqHz = freqHz;
    }

#if !NV_IS_SAFETY
    bool IsLEDEnabled(void)
    {
        return m_toggleLED_ON;
    }
    bool IsPrevFrameLEDEnabled(void)
    {
        return m_prevFrameLEDEnabled;
    }

    void EnableLEDControl(void)
    {
        m_LEDControl = true;
    }

    bool IsLEDControlEnabled(void)
    {
        return m_LEDControl;
    }
#endif // !NV_IS_SAFETY

    uint32_t m_uSensor = -1;
    INvSIPLClient::ConsumerDesc::OutputType m_outputType;

    virtual ~CNvSIPLConsumer()
    {
        Deinit();
    }

    bool IsFrameWriteComplete(void)
    {
        return m_bFrameWriteDone;
    }

    SIPLStatus OnFrameAvailable(INvSIPLClient::INvSIPLBuffer *pBuffer,
                                NvSciSyncCpuWaitContext cpuWaitContext)
    {
        auto pNvMBuffer = (INvSIPLClient::INvSIPLNvMBuffer *)pBuffer;
        if (pNvMBuffer == nullptr)
        {
            LOG_ERR("Invalid INvSIPLClient::INvSIPLNvMBuffer pointer\n");
            return NVSIPL_STATUS_ERROR;
        }
        const auto &md = pNvMBuffer->GetImageData();
        auto &EmdData = pNvMBuffer->GetImageEmbeddedData();
        LOG_INFO("EmdData.embeddedBufTopSize: %u EmdData.embeddedBufBottomSize: %u\n",
                 EmdData.embeddedBufTopSize, EmdData.embeddedBufBottomSize);

        // Send to profiler
        if (m_pProfiler != nullptr)
        {
            SIPLStatus status = m_pProfiler->ProfileFrame(pNvMBuffer);
            if (status != NVSIPL_STATUS_OK)
            {
                LOG_ERR("Frame profiling failed\n");
                return status;
            }
        }

        // --- Raw Bayer capture-for-analysis (standalone, run this FIRST) ---
        if (m_bRawCaptureEnabled && !m_bRawCaptureDone)
        {
            std::vector<uint16_t> raw16;
            int rawWidth = 0, rawHeight = 0, rawStride = 0;
            auto rawStatus = CaptureRawBayerBuffer(pNvMBuffer, cpuWaitContext,
                                                   raw16, rawWidth, rawHeight, rawStride);
            if (rawStatus != NVSIPL_STATUS_OK)
            {
                LOG_ERR("RawCapture: CaptureRawBayerBuffer failed - disabling raw capture\n");
                m_bRawCaptureDone = true; // stop retrying every frame on a hard failure
            }
            else
            {
                // --- DEBUG: raw pixel-range sanity check ---
                uint16_t minVal, maxVal;
                MinMaxU16(raw16, minVal, maxVal);
                LOG_INFO("RawCapture: raw16 %dx%d, pixel range [%u, %u]\n",
                         rawWidth, rawHeight, (unsigned)minVal, (unsigned)maxVal);

                if (m_pRawCaptureWriter == nullptr)
                {
                    m_pRawCaptureWriter.reset(new CRawCaptureWriter);
                    if (!m_pRawCaptureWriter->Init(m_sRawCaptureFilename,
                                                   rawWidth, rawHeight,
                                                   m_uRawCaptureNumFrames))
                    {
                        LOG_ERR("RawCapture: failed to init raw capture writer\n");
                        m_pRawCaptureWriter = nullptr;
                        m_bRawCaptureDone = true;
                    }
                }

                // if (m_pRawCaptureWriter != nullptr) {
                //     m_pRawCaptureWriter->WriteFrame(raw16, rawWidth, rawHeight);
                //     if (m_pRawCaptureWriter->IsDone()) {
                //         LOG_INFO("RawCapture: reached target frame count, closing capture file\n");
                //         m_pRawCaptureWriter->Deinit();
                //         m_pRawCaptureWriter = nullptr;
                //         m_bRawCaptureDone = true;
                //     }
                // }

                if (!m_bEventGenInitialized)
                {
                    LOG_INFO("EventGenerator: initializing with %d x %d\n",
                             rawWidth, rawHeight);
                    m_pEventGenerator->initialize(rawWidth, rawHeight);
                    m_bEventGenInitialized = true;

                    if (!m_sEventFilename.empty())
                    {
                        m_pEventFileWriter.reset(new CEventFileWriter);
                        if (!m_pEventFileWriter->Init(m_sEventFilename,
                                                      rawWidth, rawHeight))
                        {
                            LOG_ERR("EventGenerator: failed to init event file writer\n");
                            m_pEventFileWriter = nullptr;
                        }
                    }
                }

                double timestamp = static_cast<double>(md.frameCaptureTSC) / m_dTscFreqHz;

                // --- DEBUG: frame interval sanity check (step 3) ---
                // Remove once TSC frequency is confirmed correct.
                static double s_lastTimestamp = -1.0;
                if (s_lastTimestamp >= 0.0)
                {
                    LOG_INFO("EventGenerator: frame interval = %.4f s (expect ~1/fps)\n",
                             timestamp - s_lastTimestamp);
                }
                s_lastTimestamp = timestamp;

                // Pointer-based overload - no OpenCV dependency.
                // stridePixels = grayWidth because our capture buffers are
                // written tightly packed (we request pitch = width*bpp from
                // NvSciBufObjGetPixels ourselves - see ExtractGrayBuffer).
                int eventFrameWidth = 1280;
                int eventFrameHeight = 720;
                int eventFrameStride = 2560;
                // evsim::EventPacket packet = m_pEventGenerator->generate(
                //     raw16.data(), rawWidth, rawHeight, rawStride, timestamp);

                auto t0 = std::chrono::high_resolution_clock::now();

                evsim::EventPacket packet = m_pEventGenerator->generate(
                    raw16.data(), eventFrameWidth, eventFrameHeight, eventFrameStride, timestamp);

                LOG_INFO("EventGenerator: frame %llu produced %zu events\n",
                         (unsigned long long)packet.frameNumber, packet.events.size());

                auto t1 = std::chrono::high_resolution_clock::now();

                cout << "[Event generation]"
                    << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                           .count()
                    << "us\n";

                // if (m_pEventFileWriter != nullptr) {
                //     if (!m_pEventFileWriter->WriteEventPacket(packet)) {
                //         LOG_ERR("EventGenerator: event write failed\n");
                //     }
                // }
            }
        }

        // --- Event generation (steps 1-4) ---
        // if (m_bEventGenEnabled && m_pEventGenerator != nullptr) {

        //     std::vector<uint16_t> grayFrame;
        //     int grayWidth = 0, grayHeight = 0, grayStride = 0;
        //     auto evStatus = ExtractGrayBuffer(pNvMBuffer, cpuWaitContext,
        //                                        grayFrame, grayWidth, grayHeight, grayStride);
        //     if (evStatus != NVSIPL_STATUS_OK) {
        //         LOG_ERR("EventGenerator: ExtractGrayBuffer failed, skipping frame\n");
        //     } else {
        //         if (!m_bEventGenInitialized) {
        //             LOG_INFO("EventGenerator: initializing with %d x %d\n",
        //                       grayWidth, grayHeight);
        //             m_pEventGenerator->initialize(grayWidth, grayHeight);
        //             m_bEventGenInitialized = true;

        //             if (!m_sEventFilename.empty()) {
        //                 m_pEventFileWriter.reset(new CEventFileWriter);
        //                 if (!m_pEventFileWriter->Init(m_sEventFilename,
        //                                                grayWidth, grayHeight)) {
        //                     LOG_ERR("EventGenerator: failed to init event file writer\n");
        //                     m_pEventFileWriter = nullptr;
        //                 }
        //             }
        //         }

        //         // --- DEBUG: pixel-range sanity check (step 2) ---
        //         // Remove once mapping is confirmed correct.
        //         uint16_t minVal, maxVal;
        //         MinMaxU16(grayFrame, minVal, maxVal);
        //         LOG_INFO("EventGenerator: grayFrame %dx%d, pixel range [%u, %u]\n",
        //                   grayWidth, grayHeight, (unsigned)minVal, (unsigned)maxVal);

        //         double timestamp = static_cast<double>(md.frameCaptureTSC) / m_dTscFreqHz;

        //         // --- DEBUG: frame interval sanity check (step 3) ---
        //         // Remove once TSC frequency is confirmed correct.
        //         static double s_lastTimestamp = -1.0;
        //         if (s_lastTimestamp >= 0.0) {
        //             LOG_INFO("EventGenerator: frame interval = %.4f s (expect ~1/fps)\n",
        //                       timestamp - s_lastTimestamp);
        //         }
        //         s_lastTimestamp = timestamp;

        //         // Pointer-based overload - no OpenCV dependency.
        //         // stridePixels = grayWidth because our capture buffers are
        //         // written tightly packed (we request pitch = width*bpp from
        //         // NvSciBufObjGetPixels ourselves - see ExtractGrayBuffer).
        //         evsim::EventPacket packet = m_pEventGenerator->generate(
        //             grayFrame.data(), grayWidth, grayHeight, grayStride, timestamp);

        //         LOG_INFO("EventGenerator: frame %llu produced %zu events\n",
        //                   (unsigned long long)packet.frameNumber, packet.events.size());

        //         if (m_pEventFileWriter != nullptr) {
        //             if (!m_pEventFileWriter->WriteEventPacket(packet)) {
        //                 LOG_ERR("EventGenerator: event write failed\n");
        //             }
        //         }
        //     }
        // }

        if (m_pFrameFeeder != nullptr)
        {
            if (m_outputType == INvSIPLClient::ConsumerDesc::OutputType::ICP)
            {
                SIPLStatus status = m_pFrameFeeder->SetInputFrame(pBuffer);
                if (status != NVSIPL_STATUS_OK)
                {
                    LOG_ERR("Failed to set input frame for two-pass ISP feeder\n");
                    return status;
                }
            }
            else
            {
                SIPLStatus status = m_pFrameFeeder->DeliverOutputFrame(pBuffer);
                if (status != NVSIPL_STATUS_OK)
                {
                    LOG_ERR("Failed to send output frame to feeder for synchronization purposes\n");
                    return status;
                }
            }
        }

        // Send to file writer
        if (m_pFileWriter && (m_uFrameCounter >= m_uNumSkipFrames))
        {
            if ((m_uNumWriteFrames == -1u) || (m_uFrameCounter < (m_uNumSkipFrames + m_uNumWriteFrames)))
            {
                // create file if it isn't created
                if (!(m_uFrameCounter - m_uNumSkipFrames))
                {
                    string sFileExt;
                    NvSciBufObj bufPtr = pNvMBuffer->GetNvSciBufImage();
                    BufferAttrs bufAttrs;
                    auto status = PopulateBufAttr(bufPtr, bufAttrs);
                    if (status != NVSIPL_STATUS_OK)
                    {
                        LOG_ERR("Consumer: PopulateBufAttr failed\n");
                        return NVSIPL_STATUS_BAD_ARGUMENT;
                    }
                    if (((bufAttrs.planeColorFormats[0] > NvSciColor_LowerBound) &&
                         (bufAttrs.planeColorFormats[0] < NvSciColor_U8V8)) ||
                        ((bufAttrs.planeColorFormats[0] > NvSciColor_Float_A16) &&
                         (bufAttrs.planeColorFormats[0] < NvSciColor_UpperBound)))
                    {
                        sFileExt = ".raw";
                    }
                    else if ((bufAttrs.planeColorFormats[0] == NvSciColor_Y16) && (bufAttrs.planeCount == 1U))
                    {
                        sFileExt = ".luma";
                    }
                    else if (((bufAttrs.planeColorFormats[0] > NvSciColor_V16U16) &&
                              (bufAttrs.planeColorFormats[0] < NvSciColor_U8)) ||
                             ((bufAttrs.planeColorFormats[0] > NvSciColor_V16) &&
                              (bufAttrs.planeColorFormats[0] < NvSciColor_A8)))
                    {
                        sFileExt = ".yuv";
                    }
                    else if ((bufAttrs.planeColorFormats[0] > NvSciColor_A16Y16U16V16) &&
                             (bufAttrs.planeColorFormats[0] < NvSciColor_X6Bayer10BGGI_RGGI))
                    {
                        sFileExt = ".rgba";
                    }
                    string sFilename = m_sFilenamePrefix + "_cam_" + std::to_string(m_uSensor) + "_out_" + std::to_string((uint32_t)m_outputType) + sFileExt;
                    auto bRawOut = (m_outputType == INvSIPLClient::ConsumerDesc::OutputType::ICP);
                    status = m_pFileWriter->Init(sFilename, bRawOut);
                    if (status != NVSIPL_STATUS_OK)
                    {
                        LOG_ERR("Failed to initialize file writer\n");
                        return status;
                    }
                }
                auto status = m_pFileWriter->WriteBufferToFile(pNvMBuffer, cpuWaitContext);
                if (status != NVSIPL_STATUS_OK)
                {
                    LOG_ERR("WriteBufferToFile failed\n");
                    return status;
                }
            }
            else
            {
                m_bFrameWriteDone = true;
            }
        }

        if (m_bShowMetadata)
        {
            PrintMetadata(md);
        }

#if !NV_IS_SAFETY
        if (m_LEDControl)
        {
            SetLEDFlag(md);
        }
#endif

#if !NV_IS_SAFETY
        if (m_pComposite != nullptr)
        {
            auto status = m_pComposite->Post(m_uID, pNvMBuffer);
            if (status != NVSIPL_STATUS_OK)
            {
                LOG_ERR("Compositor post failed\n");
                return status;
            }
        }
        if (m_pMasterNvSci != nullptr)
        {
            auto status = m_pMasterNvSci->Post(m_uSensor, m_outputType, pNvMBuffer);
            if (status != NVSIPL_STATUS_OK)
            {
                LOG_ERR("Master post failed\n");
                return status;
            }
        }
#endif // !NV_IS_SAFETY

        m_uFrameCounter++;
        return NVSIPL_STATUS_OK;
    }

private:
    // --- Raw Bayer capture (no demosaic) - std::vector, no OpenCV ---
    // Use this INSTEAD of ExtractGrayBuffer when working directly with
    // ICP (sensor) output in RAW10/12/16 Bayer format. Since each pixel
    // INDEX always sees the same CFA (color filter array) position across
    // every frame, per-pixel temporal delta is still valid without
    // demosaicing.
    //
    // ASSUMPTIONS to verify with the Python analysis script:
    //   - m_outputType is ICP (raw sensor output, pre-ISP) - if you're
    //     capturing post-ISP RAW, the fence-wait logic below needs the
    //     same EOF wait as ExtractGrayBuffer's non-ICP branch.
    //   - bufAttrs.planeColorFormats[0] is in the Bayer RAW range
    //     (matches CFileWriter's "RAW" branch condition).
    //   - Container is 16-bit (bpp == 2) even if sensor bit depth is
    //     10/12-bit - values will be left- or right-justified within the
    //     16 bits depending on sensor/ISP config, which the Python script
    //     checks via max-value histogram.
    //
    // Since NvSciBufObjGetPixels can be told to write at whatever pitch we
    // request, we request pitch == width*2 (tightly packed) and write
    // straight into the destination vector's own storage - no separate
    // scratch buffer or copy needed, since the buffer is already native
    // 16-bit (no widening from 8-bit required, unlike ExtractGrayBuffer).
    SIPLStatus CaptureRawBayerBuffer(INvSIPLClient::INvSIPLNvMBuffer *pNvMBuffer,
                                     NvSciSyncCpuWaitContext cpuWaitContext,
                                     std::vector<uint16_t> &outRaw16,
                                     int &outWidth, int &outHeight, int &outStridePixels)
    {
        NvSciError sciErr;
        SIPLStatus status;

        if (m_outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP)
        {
            NvSciSyncFence fence = NvSciSyncFenceInitializer;
            status = pNvMBuffer->GetEOFNvSciSyncFence(&fence);
            if (status != NVSIPL_STATUS_OK)
            {
                LOG_ERR("RawCapture: GetEOFNvSciSyncFence failed\n");
                return status;
            }
            sciErr = NvSciSyncFenceWait(&fence, cpuWaitContext, FENCE_FRAME_TIMEOUT_MS * 1000UL);
            NvSciSyncFenceClear(&fence);
            if (sciErr != NvSciError_Success)
            {
                LOG_ERR("RawCapture: NvSciSyncFenceWait failed (err=%d)\n", (int)sciErr);
                return NVSIPL_STATUS_ERROR;
            }
        }

        NvSciBufObj bufPtr = pNvMBuffer->GetNvSciBufImage();
        BufferAttrs bufAttrs;
        status = PopulateBufAttr(bufPtr, bufAttrs);

        LOG_INFO("Plane count : %d\n", bufAttrs.planeCount);
        for (int i = 0; i < bufAttrs.planeCount; i++)
        {
            LOG_INFO("Color Format : %d\n", bufAttrs.planeColorFormats[i]);
            LOG_INFO("Bits per pixel : %d\n", bufAttrs.planeBitsPerPixels[i]);
            LOG_INFO("Plane Width : %d plane Height : %d\n", bufAttrs.planeWidths[i], bufAttrs.planeHeights[i]);
            LOG_INFO("Plane Pitch : %d\n", bufAttrs.planePitches[i]);
        }

        if (status != NVSIPL_STATUS_OK)
        {
            LOG_ERR("RawCapture: PopulateBufAttr failed\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        // Bayer RAW range check - mirrors the condition CFileWriter uses
        // to pick its "RAW" branch in GetBuffParams(). Log the actual
        // enum value on mismatch so we can adjust this range if your
        // pipeline reports something outside it.
        bool isBayerRaw =
            (1U == bufAttrs.planeCount) &&
            ((bufAttrs.planeColorFormats[0] < NvSciColor_U8V8) ||
             (bufAttrs.planeColorFormats[0] == NvSciColor_X4Bayer12RGGB_RJ) ||
             ((bufAttrs.planeColorFormats[0] >= NvSciColor_X6Bayer10BGGI_RGGI) &&
              (bufAttrs.planeColorFormats[0] <= NvSciColor_Bayer16IGGR_IGGB)));

        if (!isBayerRaw)
        {
            LOG_ERR("RawCapture: planeColorFormats[0]=%u is not in the expected "
                    "Bayer RAW range - update isBayerRaw check\n",
                    (unsigned)bufAttrs.planeColorFormats[0]);
            return NVSIPL_STATUS_NOT_SUPPORTED;
        }

        uint32_t bpp = 0U;
        if (!CUtils::GetBpp(bufAttrs.planeBitsPerPixels[0], &bpp))
        {
            LOG_ERR("RawCapture: GetBpp failed\n");
            return NVSIPL_STATUS_ERROR;
        }
        if (bpp != 2U)
        {
            LOG_ERR("RawCapture: expected 16-bit container (bpp=2), got bpp=%u\n", bpp);
            return NVSIPL_STATUS_NOT_SUPPORTED;
        }

        uint32_t width = bufAttrs.planeWidths[0];
        uint32_t height = bufAttrs.planeHeights[0];
        uint32_t pitch = width * bpp; // tightly packed - request this exact pitch
        uint32_t size = pitch * height;

        outRaw16.resize(static_cast<size_t>(width) * height);

        if (bufAttrs.needSwCacheCoherency)
        {
            sciErr = NvSciBufObjFlushCpuCacheRange(bufPtr, 0U, bufAttrs.planePitches[0] * height);
            if (sciErr != NvSciError_Success)
            {
                LOG_ERR("RawCapture: NvSciBufObjFlushCpuCacheRange failed\n");
                return NVSIPL_STATUS_ERROR;
            }
        }

        void *pPixels[1] = {outRaw16.data()};
        uint32_t sizes[1] = {size};
        uint32_t pitches[1] = {pitch};

        sciErr = NvSciBufObjGetPixels(bufPtr, nullptr, pPixels, sizes, pitches);
        if (sciErr != NvSciError_Success)
        {
            LOG_ERR("RawCapture: NvSciBufObjGetPixels failed (err=%d)\n", (int)sciErr);
            return NVSIPL_STATUS_ERROR;
        }

        outWidth = static_cast<int>(width);
        outHeight = static_cast<int>(height);
        outStridePixels = static_cast<int>(width); // tightly packed, no row padding

        return NVSIPL_STATUS_OK;
    }

    // --- Generalized single-plane capture: Bayer, Luma, RGBA, packed YUV ---
    //
    // Covers the SAME format buckets the original consumer's file-extension
    // logic detects (.raw/.luma/.rgba/ and single-plane .yuv), but returns
    // raw bytes + metadata instead of writing to disk - use this when you
    // need the frame for further C++ processing (event generation, CV ops)
    // rather than just archiving bytes (for pure archiving, CFileWriter
    // already does this generically - see WriteBufferToFile).
    //
    // NOT covered yet: semi-planar YUV (2-3 planes, e.g. NV12) - that needs
    // separate multi-plane handling (each plane has its own pitch, and
    // combining them for a later NV12->BGR conversion needs a specific
    // stacked layout). Returns NVSIPL_STATUS_NOT_SUPPORTED for that case
    // for now, flagged explicitly rather than guessed at.
    enum class FrameFormatBucket
    {
        BAYER_RAW,      // Bayer mosaic, any bit depth/CFA pattern
        LUMA,           // single-plane Y8/Y10/Y12/Y16
        RGBA,           // packed RGBA/BGRA (8-bit) or float RGBA
        YUV_PACKED,     // single-plane packed YUV (e.g. A8Y8U8V8)
        YUV_SEMIPLANAR, // 2-3 plane YUV (e.g. NV12) - NOT YET SUPPORTED here
        UNKNOWN
    };

    FrameFormatBucket ClassifyFormat(const BufferAttrs &bufAttrs)
    {
        auto fmt = bufAttrs.planeColorFormats[0];

        if (bufAttrs.planeCount >= 2U &&
            ((fmt == NvSciColor_Y8) || (fmt == NvSciColor_Y10) ||
             (fmt == NvSciColor_Y12) || (fmt == NvSciColor_Y16)))
        {
            return FrameFormatBucket::YUV_SEMIPLANAR;
        }

        if ((bufAttrs.planeCount == 1U) &&
            ((fmt == NvSciColor_Y8) || (fmt == NvSciColor_Y10) ||
             (fmt == NvSciColor_Y12) || (fmt == NvSciColor_Y16)))
        {
            return FrameFormatBucket::LUMA;
        }

        if ((fmt >= NvSciColor_A8Y8U8V8) && (fmt <= NvSciColor_A16Y16U16V16))
        {
            return FrameFormatBucket::YUV_PACKED;
        }

        if ((fmt == NvSciColor_Float_A16B16G16R16) ||
            ((fmt >= NvSciColor_B8G8R8A8) && (fmt <= NvSciColor_A8B8G8R8)))
        {
            return FrameFormatBucket::RGBA;
        }

        // Bayer range - mirrors CaptureRawBayerBuffer's isBayerRaw check
        if ((bufAttrs.planeCount == 1U) &&
            ((fmt < NvSciColor_U8V8) ||
             (fmt == NvSciColor_X4Bayer12RGGB_RJ) ||
             ((fmt >= NvSciColor_X6Bayer10BGGI_RGGI) &&
              (fmt <= NvSciColor_Bayer16IGGR_IGGB))))
        {
            return FrameFormatBucket::BAYER_RAW;
        }

        return FrameFormatBucket::UNKNOWN;
    }

    // Generic single-plane extractor: works for BAYER_RAW, LUMA, RGBA,
    // YUV_PACKED. Returns raw bytes + the metadata needed to interpret
    // them (bytesPerPixel, width, height) - no OpenCV, caller reinterprets
    // as needed (e.g. cast outBytes.data() to uint16_t* for bpp==2).
    // Caller checks ClassifyFormat() first; this function trusts outBucket
    // and does not re-validate the format range.
    SIPLStatus CaptureFrameGeneric(INvSIPLClient::INvSIPLNvMBuffer *pNvMBuffer,
                                   NvSciSyncCpuWaitContext cpuWaitContext,
                                   std::vector<uint8_t> &outBytes,
                                   int &outWidth, int &outHeight, int &outBytesPerPixel,
                                   FrameFormatBucket &outBucket)
    {
        NvSciError sciErr;
        SIPLStatus status;

        if (m_outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP)
        {
            NvSciSyncFence fence = NvSciSyncFenceInitializer;
            status = pNvMBuffer->GetEOFNvSciSyncFence(&fence);
            if (status != NVSIPL_STATUS_OK)
            {
                LOG_ERR("CaptureFrameGeneric: GetEOFNvSciSyncFence failed\n");
                return status;
            }
            sciErr = NvSciSyncFenceWait(&fence, cpuWaitContext, FENCE_FRAME_TIMEOUT_MS * 1000UL);
            NvSciSyncFenceClear(&fence);
            if (sciErr != NvSciError_Success)
            {
                LOG_ERR("CaptureFrameGeneric: NvSciSyncFenceWait failed (err=%d)\n", (int)sciErr);
                return NVSIPL_STATUS_ERROR;
            }
        }

        NvSciBufObj bufPtr = pNvMBuffer->GetNvSciBufImage();
        BufferAttrs bufAttrs;
        status = PopulateBufAttr(bufPtr, bufAttrs);
        if (status != NVSIPL_STATUS_OK)
        {
            LOG_ERR("CaptureFrameGeneric: PopulateBufAttr failed\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        outBucket = ClassifyFormat(bufAttrs);
        if (outBucket == FrameFormatBucket::YUV_SEMIPLANAR)
        {
            LOG_ERR("CaptureFrameGeneric: semi-planar YUV not yet supported "
                    "(needs multi-plane handling) - see note in code\n");
            return NVSIPL_STATUS_NOT_SUPPORTED;
        }
        if (outBucket == FrameFormatBucket::UNKNOWN)
        {
            LOG_ERR("CaptureFrameGeneric: unrecognized format (planeColorFormats[0]=%u)\n",
                    (unsigned)bufAttrs.planeColorFormats[0]);
            return NVSIPL_STATUS_NOT_SUPPORTED;
        }

        uint32_t bpp = 0U;
        if (!CUtils::GetBpp(bufAttrs.planeBitsPerPixels[0], &bpp))
        {
            LOG_ERR("CaptureFrameGeneric: GetBpp failed\n");
            return NVSIPL_STATUS_ERROR;
        }

        // RGBA/float-RGBA bytesPerPixel isn't derived from planeBitsPerPixels
        // the same way (CFileWriter hardcodes 4 or 8 for these) - match that.
        if (outBucket == FrameFormatBucket::RGBA)
        {
            bpp = (bufAttrs.planeColorFormats[0] == NvSciColor_Float_A16B16G16R16) ? 8U : 4U;
        }

        uint32_t width = bufAttrs.planeWidths[0];
        uint32_t height = bufAttrs.planeHeights[0];
        uint32_t pitch = width * bpp; // tightly packed - request this exact pitch
        uint32_t size = pitch * height;

        outBytes.resize(size);

        if (bufAttrs.needSwCacheCoherency)
        {
            sciErr = NvSciBufObjFlushCpuCacheRange(bufPtr, 0U, bufAttrs.planePitches[0] * height);
            if (sciErr != NvSciError_Success)
            {
                LOG_ERR("CaptureFrameGeneric: NvSciBufObjFlushCpuCacheRange failed\n");
                return NVSIPL_STATUS_ERROR;
            }
        }

        void *pPixels[1] = {outBytes.data()};
        uint32_t sizes[1] = {size};
        uint32_t pitches[1] = {pitch};

        sciErr = NvSciBufObjGetPixels(bufPtr, nullptr, pPixels, sizes, pitches);
        if (sciErr != NvSciError_Success)
        {
            LOG_ERR("CaptureFrameGeneric: NvSciBufObjGetPixels failed (err=%d)\n", (int)sciErr);
            return NVSIPL_STATUS_ERROR;
        }

        // NOTE: for YUV_PACKED, outBytes is an opaque N-byte-per-pixel blob
        // without interpreting chroma subsampling/ordering (YUYV vs UYVY vs
        // A8Y8U8V8 all differ) - decode downstream once we know the exact
        // packed layout in use.
        outWidth = static_cast<int>(width);
        outHeight = static_cast<int>(height);
        outBytesPerPixel = static_cast<int>(bpp);

        return NVSIPL_STATUS_OK;
    }

    // --- Gray extraction (Y8/Y10/Y12/Y16 luma) - std::vector, no OpenCV ---
    // Delivers native pixel values as uint16_t (8-bit widened, 16-bit as-is)
    // rather than downscaling everything to 8-bit - matches EventGenerator's
    // uint16_t pointer-based generate() overload directly.
    SIPLStatus ExtractGrayBuffer(INvSIPLClient::INvSIPLNvMBuffer *pNvMBuffer,
                                 NvSciSyncCpuWaitContext cpuWaitContext,
                                 std::vector<uint16_t> &outGray,
                                 int &outWidth, int &outHeight, int &outStridePixels)
    {
        NvSciError sciErr;
        SIPLStatus status;

        // Wait on EOF fence for ISP output, matching CFileWriter's convention
        // (ICP/raw output doesn't need a wait).
        if (m_outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP)
        {
            NvSciSyncFence fence = NvSciSyncFenceInitializer;
            status = pNvMBuffer->GetEOFNvSciSyncFence(&fence);
            if (status != NVSIPL_STATUS_OK)
            {
                LOG_ERR("EventGenerator: GetEOFNvSciSyncFence failed\n");
                return status;
            }
            sciErr = NvSciSyncFenceWait(&fence, cpuWaitContext, FENCE_FRAME_TIMEOUT_MS * 1000UL);
            NvSciSyncFenceClear(&fence);
            if (sciErr != NvSciError_Success)
            {
                LOG_ERR("EventGenerator: NvSciSyncFenceWait failed (err=%d)\n", (int)sciErr);
                return NVSIPL_STATUS_ERROR;
            }
        }

        NvSciBufObj bufPtr = pNvMBuffer->GetNvSciBufImage();
        BufferAttrs bufAttrs;
        status = PopulateBufAttr(bufPtr, bufAttrs);
        if (status != NVSIPL_STATUS_OK)
        {
            LOG_ERR("EventGenerator: PopulateBufAttr failed\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        uint32_t bpp = 0U;
        if ((bufAttrs.planeColorFormats[0] == NvSciColor_Y8) ||
            (bufAttrs.planeColorFormats[0] == NvSciColor_Y10) ||
            (bufAttrs.planeColorFormats[0] == NvSciColor_Y12) ||
            (bufAttrs.planeColorFormats[0] == NvSciColor_Y16))
        {
            if (!CUtils::GetBpp(bufAttrs.planeBitsPerPixels[0], &bpp))
            {
                LOG_ERR("EventGenerator: GetBpp failed\n");
                return NVSIPL_STATUS_ERROR;
            }
        }
        else
        {
            LOG_ERR("EventGenerator: unsupported pixel format (%u) for gray extraction\n",
                    (unsigned)bufAttrs.planeColorFormats[0]);
            return NVSIPL_STATUS_NOT_SUPPORTED;
        }

        uint32_t width = bufAttrs.planeWidths[0];
        uint32_t height = bufAttrs.planeHeights[0];

        if (bufAttrs.needSwCacheCoherency)
        {
            NvSciError flushErr = NvSciBufObjFlushCpuCacheRange(
                bufPtr, 0U, bufAttrs.planePitches[0] * height);
            if (flushErr != NvSciError_Success)
            {
                LOG_ERR("EventGenerator: NvSciBufObjFlushCpuCacheRange failed\n");
                return NVSIPL_STATUS_ERROR;
            }
        }

        outGray.resize(static_cast<size_t>(width) * height);

        if (bpp == 2U)
        {
            // Native 16-bit - request tightly-packed pitch and write
            // straight into the destination vector, no intermediate copy.
            uint32_t pitch = width * bpp;
            uint32_t size = pitch * height;

            void *pPixels[1] = {outGray.data()};
            uint32_t sizes[1] = {size};
            uint32_t pitches[1] = {pitch};

            sciErr = NvSciBufObjGetPixels(bufPtr, nullptr, pPixels, sizes, pitches);
            if (sciErr != NvSciError_Success)
            {
                LOG_ERR("EventGenerator: NvSciBufObjGetPixels failed (err=%d)\n", (int)sciErr);
                return NVSIPL_STATUS_ERROR;
            }
        }
        else if (bpp == 1U)
        {
            // 8-bit source: capture into a reusable byte scratch buffer,
            // then widen into the uint16_t destination (no scaling - just
            // a straight value copy, e.g. 200 -> 200, not 200 -> 51200).
            uint32_t pitch = width * bpp;
            uint32_t size = pitch * height;

            m_grayScratch8.resize(size);

            void *pPixels[1] = {m_grayScratch8.data()};
            uint32_t sizes[1] = {size};
            uint32_t pitches[1] = {pitch};

            sciErr = NvSciBufObjGetPixels(bufPtr, nullptr, pPixels, sizes, pitches);
            if (sciErr != NvSciError_Success)
            {
                LOG_ERR("EventGenerator: NvSciBufObjGetPixels failed (err=%d)\n", (int)sciErr);
                return NVSIPL_STATUS_ERROR;
            }

            for (size_t i = 0; i < m_grayScratch8.size(); ++i)
            {
                outGray[i] = static_cast<uint16_t>(m_grayScratch8[i]);
            }
        }
        else
        {
            LOG_ERR("EventGenerator: unexpected bytesPerPixel %u\n", bpp);
            return NVSIPL_STATUS_NOT_SUPPORTED;
        }

        outWidth = static_cast<int>(width);
        outHeight = static_cast<int>(height);
        outStridePixels = static_cast<int>(width); // tightly packed, no row padding

        return NVSIPL_STATUS_OK;
    }

    void PrintMetadata(const INvSIPLClient::ImageMetaData &md)
    {
        cout << "Camera ID: " << m_uSensor << endl;
        cout << " Frame Counter: " << (md.frameSeqNumInfo.frameSeqNumValid ? md.frameSeqNumInfo.frameSequenceNumber : m_uFrameCounter)
             << endl;
        cout << " TSC SOF: " << md.frameCaptureStartTSC << endl;
        cout << " TSC EOF: " << md.frameCaptureTSC << endl;
        if (md.badPixelStatsValid)
        {
            cout << " Bad pixel stats:" << endl;
            cout << "     highInWin: " << md.badPixelStats.highInWin << endl;
            cout << "     lowInWin: " << md.badPixelStats.lowInWin << endl;
            cout << "     highMagInWin: " << md.badPixelStats.highMagInWin << endl;
            cout << "     lowMagInWin: " << md.badPixelStats.lowMagInWin << endl;
            cout << "     highOutWin: " << md.badPixelStats.highOutWin << endl;
            cout << "     lowOutWin: " << md.badPixelStats.lowOutWin << endl;
            cout << "     highMagOutWin: " << md.badPixelStats.highMagOutWin << endl;
            cout << "     lowMagOutWin: " << md.badPixelStats.lowMagOutWin << endl;

            cout << " Bad pixel settings:" << endl;
            cout << "     ROI: " << md.badPixelSettings.rectangularMask.x0 << ", "
                 << md.badPixelSettings.rectangularMask.x1 << ", "
                 << md.badPixelSettings.rectangularMask.y0 << ", "
                 << md.badPixelSettings.rectangularMask.y1 << endl;
        }

        if (md.controlInfo.valid)
        {
            cout << "alpha: " << md.controlInfo.alpha << endl;
            if (md.controlInfo.isLuminanceCalibrated)
            {
                cout << "luminanceCalibrationFactor: " << md.controlInfo.luminanceCalibrationFactor << endl;
            }
            else
            {
                cout << "luminance is not calibrated" << endl;
            }
            if (md.controlInfo.wbGainTotal.valid)
            {
                cout << "wbGains[0]: " << md.controlInfo.wbGainTotal.gain[0] << endl;
                cout << "wbGains[1]: " << md.controlInfo.wbGainTotal.gain[1] << endl;
                cout << "wbGains[2]: " << md.controlInfo.wbGainTotal.gain[2] << endl;
                cout << "wbGains[3]: " << md.controlInfo.wbGainTotal.gain[3] << endl;
            }
            else
            {
                cout << "wbGain info not enabled" << endl;
            }
            cout << "cct: " << md.controlInfo.cct << endl;
            cout << "brightnessKey: " << md.controlInfo.brightnessKey << endl;
            cout << "rawImageMidTone: " << md.controlInfo.rawImageMidTone << endl;
            if (md.controlInfo.gtmSplineInfo.enable)
            {
                cout << "gtmSpline control points: " << endl;
                for (uint32_t i = 0U; i < NUM_GTM_SPLINE_POINTS; i++)
                {
                    cout << "    index " << i << " -- x: " << md.controlInfo.gtmSplineInfo.gtmSplineControlPoint[i].x << " y: " << md.controlInfo.gtmSplineInfo.gtmSplineControlPoint[i].y << " slope: " << md.controlInfo.gtmSplineInfo.gtmSplineControlPoint[i].slope << endl;
                }
            }
            else
            {
                cout << "gtmSpline info not enabled" << endl;
            }
        }
        else
        {
            cout << "controlInfo not valid" << endl;
        }

        for (uint32_t i = 0; i < 2; i++)
        {
            if (md.histogramStatsValid[i])
            {
                cout << " Histogram[" << i << "] stats:" << endl;
                for (uint32_t comp = 0; comp < NVSIPL_ISP_MAX_COLOR_COMPONENT; comp++)
                {
                    cout << "     data[][" << comp << "] (first 8 values): ";
                    for (uint32_t j = 0; j < 8; j++)
                    {
                        cout << md.histogramStats[i].data[j][comp] << " ";
                    }
                    cout << endl;
                }

                cout << " Histogram[" << i << "] settings:" << endl;
                cout << "     knees[]: ";
                for (uint32_t point = 0; point < NVSIPL_ISP_HIST_KNEE_POINTS; point++)
                {
                    cout << unsigned(md.histogramSettings[i].knees[point]) << " ";
                }
                cout << endl;

                cout << "     ranges[]: ";
                for (uint32_t point = 0; point < NVSIPL_ISP_HIST_KNEE_POINTS; point++)
                {
                    cout << unsigned(md.histogramSettings[i].ranges[point]) << " ";
                }
                cout << endl;
            }
        }

        for (uint32_t i = 0; i < 2; i++)
        {
            if (md.localAvgClipStatsValid[i])
            {
                cout << " LocalAvgClip[" << i << "] stats:" << endl;
                for (uint32_t roi = 0; roi < NVSIPL_ISP_MAX_LAC_ROI; roi++)
                {
                    cout << "     data[" << roi << "].numWindowsH: " << md.localAvgClipStats[i].data[0].numWindowsH << endl;
                    cout << "     data[" << roi << "].numWindowsV: " << md.localAvgClipStats[i].data[0].numWindowsH << endl;
                    for (uint32_t comp = 0; comp < NVSIPL_ISP_MAX_COLOR_COMPONENT; comp++)
                    {
                        cout << "     data[" << roi << "].average[][" << comp << "] (first 8 values): ";
                        for (uint32_t j = 0; j < 8; j++)
                        {
                            cout << md.localAvgClipStats[i].data[roi].average[j][comp] << " ";
                        }
                        cout << endl;
                    }
                }

                cout << " LocalAvgClip[" << i << "] settings:" << endl;
                for (uint32_t roi = 0; roi < NVSIPL_ISP_MAX_LAC_ROI; roi++)
                {
                    cout << "     windows[" << roi << "].width: " << md.localAvgClipSettings[i].windows[roi].width << endl;
                    cout << "     windows[" << roi << "].height: " << md.localAvgClipSettings[i].windows[roi].height << endl;
                    cout << "     windows[" << roi << "].numWindowsH: " << md.localAvgClipSettings[i].windows[roi].numWindowsH << endl;
                    cout << "     windows[" << roi << "].numWindowsV: " << md.localAvgClipSettings[i].windows[roi].numWindowsV << endl;
                    cout << "     windows[" << roi << "].horizontalInterval: " << md.localAvgClipSettings[i].windows[roi].horizontalInterval << endl;
                    cout << "     windows[" << roi << "].verticalInterval: " << md.localAvgClipSettings[i].windows[roi].verticalInterval << endl;
                    cout << "     windows[" << roi << "].startOffset: (" << md.localAvgClipSettings[i].windows[roi].startOffset.x << ", "
                         << md.localAvgClipSettings[i].windows[roi].startOffset.y << ")" << endl;
                }
            }
        }
#if !NV_IS_SAFETY
        if (md.frameTimestampInfo.frameTimestampValid)
        {
            cout << " Frame Timestamp from the sensor: " << md.frameTimestampInfo.frameTimestamp << endl;
        }
#endif //! NV_IS_SAFETY

        cout << " errorFlag: " << (int)md.errorFlag << " (meaning determined by driver)" << endl;
    }

#if !NV_IS_SAFETY
    void SetLEDFlag(const INvSIPLClient::ImageMetaData &md)
    {
        if (md.sensorExpInfo.expTimeValid)
        {
            constexpr double_t MAX_EXP_THRESHOLD_AR0234 = 0.0013; // 1.3ms
            constexpr double_t MIN_EXP_THRESHOLD_AR0234 = 0.0004; // 0.4ms

            m_prevFrameLEDEnabled = IsLEDEnabled();
            if (md.sensorExpInfo.exposureTime[0] >= MAX_EXP_THRESHOLD_AR0234)
            {
                m_toggleLED_ON = true; // turn on LED when exposure hits max threshold
            }
            if (md.sensorExpInfo.exposureTime[0] <= MIN_EXP_THRESHOLD_AR0234)
            {
                m_toggleLED_ON = false; // turn off LED when exposure hits min threshold
            }
        }
    }
#endif //! NV_IS_SAFETY

    unique_ptr<CFileWriter> m_pFileWriter = nullptr;
    string m_sFilenamePrefix = "";
    uint32_t m_uNumSkipFrames = 0u;
    uint64_t m_uNumWriteFrames = -1u;
    bool m_bFrameWriteDone = false;
#if !NV_IS_SAFETY
    CComposite *m_pComposite = nullptr;
    CNvSIPLMasterNvSci *m_pMasterNvSci = nullptr;
    bool m_prevFrameLEDEnabled = true;
    bool m_toggleLED_ON = true;
    bool m_LEDControl = false;
#endif // !NV_IS_SAFETY
    uint32_t m_uID = -1;
    uint32_t m_uFrameCounter = 0u;
    CProfiler *m_pProfiler = nullptr;
    CFrameFeeder *m_pFrameFeeder = nullptr;
    bool m_bShowMetadata = false;

    // --- Event generation members (steps 1-4) ---
    unique_ptr<evsim::EventGenerator> m_pEventGenerator = nullptr;
    bool m_bEventGenEnabled = false;
    bool m_bEventGenInitialized = false;

    unique_ptr<CEventFileWriter> m_pEventFileWriter = nullptr;
    string m_sEventFilename = "";

    // Scratch buffer for the bpp==1 (8-bit luma) case in ExtractGrayBuffer,
    // reused across frames to avoid per-frame allocation.
    std::vector<uint8_t> m_grayScratch8;

    // TSC ticks-per-second - ASSUMPTION, verify against platform docs (step 3)
    double m_dTscFreqHz = 31250000.0;

    // --- Raw Bayer capture-for-analysis members ---
    unique_ptr<CRawCaptureWriter> m_pRawCaptureWriter = nullptr;
    string m_sRawCaptureFilename = "";
    uint32_t m_uRawCaptureNumFrames = 10u;
    bool m_bRawCaptureEnabled = false;
    bool m_bRawCaptureDone = false;

    // Helper: min/max over a uint16_t buffer, replaces cv::minMaxLoc for
    // the debug sanity-check logs (no OpenCV dependency).
    static void MinMaxU16(const std::vector<uint16_t> &v, uint16_t &outMin, uint16_t &outMax)
    {
        if (v.empty())
        {
            outMin = 0;
            outMax = 0;
            return;
        }
        outMin = v[0];
        outMax = v[0];
        for (uint16_t val : v)
        {
            if (val < outMin)
                outMin = val;
            if (val > outMax)
                outMax = val;
        }
    }
};

#endif // CNVSIPLCONSUMER_HPP
