/////////////////////////////////////////////////////////////////////////////////////////
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA CORPORATION & AFFILIATES assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA CORPORATION & AFFILIATES. No third party distribution is allowed unless
// expressly authorized by NVIDIA. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA CORPORATION & AFFILIATES products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA CORPORATION & AFFILIATES.
//
// SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <lodepng.h>

// evsim: CUDA event-camera simulator
#include "Config.h"
#include "EventPacket.h"
#include "IEventGenerator.h"
#include "EventGeneratorCUDA.cuh"

#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/image/Image.h>
#include <dw/interop/streamer/ImageStreamer.h>
#include <dw/rig/Rig.h>
#include <dw/sensors/codecs/Codec.h>
#include <dw/sensors/codecs/CodecHeader.h>
#include <dw/sensors/common/SensorSerializer.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/camera/Camera.h>
#include <dw/sensors/camera/CodecHeaderVideo.h>
#include <dw/sensors/codecs/sensorserializer/SensorSerializer.h>

#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Renderer.h>
#include <dwvisualization/core/Visualization.h>
#include <dwvisualization/image/Image.h>
#include <dwvisualization/interop/ImageStreamer.h>

#include <framework/Checks.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/CudaTimer.hpp>
#include <framework/Log.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/RenderUtils.hpp>
#include <framework/SamplesDataPath.hpp>
#include <framework/ScreenshotHelper.hpp>
#include <framework/WindowGLFW.hpp>

#include <algorithm>

using namespace dw_samples::common;

#define MAX_PORTS_COUNT 4 // the number of device blocks, each with multiple cameras
#define MAX_CAMS_PER_PORT 4
#define MAX_CAMS MAX_PORTS_COUNT *MAX_CAMS_PER_PORT
#define FIRST_CAMERA_IDX 0
#define MAX_PIPELINE_COUNT 4 // ICP/ISP0/ISP1/ISP2
#define ICP_PIPELINE_IDX 0
#define ISP_PIPELINE_IDX 1
#define ISP1_PIPELINE_IDX 2
#define ISP2_PIPELINE_IDX 3
#define MAX_CAMERA_FIFO_SIZE 30

bool isResetLinkOnError = false;

// declaration for error handling
void cameraEventHandling(dwCameraSIPLNotification *notificationData, dwSensorHandle_t sensor);
auto &cameraDevBlkEventHandling = cameraEventHandling;
auto &cameraPipelineEventHandling = cameraEventHandling;

const char *PATTERNYUV420BL1 = "isp-mode=yuv420-uint8-bl";
const char *PATTERNYUV420BL2 = "isp-mode=yuv420-uint16-bl";

class CameraCustomSimpleApp : public DriveWorksSample
{
private:
    // Driveworks Context and SAL
    dwContextHandle_t m_context = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t m_viz = DW_NULL_HANDLE;
    dwRendererHandle_t m_renderer = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    dwRenderEngineParams params{};
    dwRenderEngineColorRGBA m_colorPerPort[MAX_PORTS_COUNT];

    dwSALHandle_t m_sal = DW_NULL_HANDLE;
    dwSensorHandle_t m_cameraMaster = DW_NULL_HANDLE;
    uint32_t m_totalCameras; // actual number of cameras used
    bool m_isProcessed1WriteEnabled;
    bool m_bUseNewCodecStack;

    dwRigHandle_t m_rigConfig{};

    // Per camera.
    // All active objects are at the front of array in order of the port.
    // So if there number of cameras per port is [4,2,4,1] then the
    // first 11 elements of these arrays will contain their information.
    // The total number of cameras is stored in m_totalCameras
    dwSensorHandle_t m_camera[MAX_CAMS];
    dwImageStreamerHandle_t m_streamerToGL[MAX_CAMS] = {DW_NULL_HANDLE};
    uint32_t m_tileVideo[MAX_CAMS + 1];
    dwRectf m_renderRanges[MAX_CAMS] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    bool m_enableRender[MAX_CAMS] = {true};

    bool m_recordCamera = false;
    bool m_recordRaw = false;
    dwSensorSerializerHandle_t m_serializer[dwCameraISPType::DW_CAMERA_MAX_ISP_COUNT] = {DW_NULL_HANDLE, DW_NULL_HANDLE, DW_NULL_HANDLE};
    dwSensorSerializerNewHandle_t m_serializerNew[dwCameraISPType::DW_CAMERA_MAX_ISP_COUNT] = {DW_NULL_HANDLE, DW_NULL_HANDLE, DW_NULL_HANDLE};
    dwCodecHeaderHandle_t m_codecHeader = DW_NULL_HANDLE;

    bool m_useRaw = false;
    bool m_useProcessed = false;
    bool m_useProcessed1 = false;
    bool m_useProcessed2 = false;
    uint32_t m_fifoSize = 0U;

    std::unique_ptr<ScreenshotHelper> m_screenshot;

    // Frame grab variables
    dwRect m_roi;
    std::string m_pathName;
    dwImageStreamerHandle_t m_streamerToCPUGrab[MAX_CAMS] = {DW_NULL_HANDLE};
    int m_screenshotCount = 0;
    bool m_frameGrab = false;
    uint64_t m_frameCount = 0;

    // press P, then group 0-3 then link 0-3 and that camera will be stopped
    bool m_pauseSensor = false;
    // press O, then group 0-3 then link 0-3 and that camera will be started
    bool m_startSensor = false;
    // press I, then group 0-3 then link 0-3 and that camera will be reset
    bool m_resetSensor = false;

    bool m_isCameraSelected = false;
    uint32_t m_cameraSelected = 0;

    // List of imageHandle activated with setImagePool
    // std::vector<dwImageHandle_t> m_imagePoolList;
    dwImageHandle_t m_imagesForImagePool[MAX_CAMS][MAX_PIPELINE_COUNT][MAX_CAMERA_FIFO_SIZE] = {nullptr};

    enum RecordFileFormat
    {
        RECORD_FILE_FORMAT_H264,
        RECORD_FILE_FORMAT_H265,
        RECORD_FILE_FORMAT_MP4,
        RECORD_FILE_FORMAT_LRAW,
        RECORD_FILE_FORMAT_RAW,
        RECORD_FILE_FORMAT_POD,
        RECORD_FILE_FORMAT_UNSUPPORTED,
    };

    RecordFileFormat m_recordFormat = RECORD_FILE_FORMAT_UNSUPPORTED;

    // ---- evsim event generation ----
    // One generator, attached to a single camera (--event-camera). Two inputs:
    //   raw  (default): DW_CAMERA_OUTPUT_NATIVE_RAW is streamed to a CUDA image
    //                   and handed to generateFromDevice() -- no grayscale
    //                   conversion, no host copy. Needs "raw" in the rig's
    //                   output-requested.
    //   rgba          : CUDA RGBA8 (processed) image -> host luma -> generate().
    std::unique_ptr<evsim::EventGeneratorCUDA> m_eventGen;
    bool m_eventUseRaw = true;
    int m_eventRawShift = 0;
    dwImageStreamerHandle_t m_streamerRawToCUDA = DW_NULL_HANDLE;
    uint32_t m_eventCamera = 0;
    std::vector<uint16_t> m_lumaBuf;
    std::ofstream m_eventCsv;
    uint64_t m_eventFrames = 0;
    uint64_t m_eventTotal = 0;
    size_t m_eventLastPacket = 0;

    // ---- event display (dummy visualisation) ----
    bool m_eventDisplay = false; // extra tile showing the last packet
    uint32_t m_eventW = 0;
    uint32_t m_eventH = 0;
    // size_t m_eventDrawCap = 100000; // max points drawn per colour per frame
    size_t m_eventDrawCap = 30000;
    std::vector<dwVector2f> m_evOn;
    std::vector<dwVector2f> m_evOff;
    std::vector<uint16_t> m_evOnCnt, m_evOffCnt;               // W*H counters (replaces m_evOn/m_evOff)
    dwImageHandle_t m_evImg = DW_NULL_HANDLE;                  // CPU RGBA8 image
    dwImageStreamerHandle_t m_streamerEvToGL = DW_NULL_HANDLE; // CPU -> GL
    int m_evFramesPerWin = 1;                                  // accumulate this many frames per refresh
    int m_evFrameCount = 0;
    float m_evGain = 64.0f; // counts -> 0..255; 1 count = 64 brightness

    // uint32_t m_tileVideo[MAX_CAMS + 1];

public:
    CameraCustomSimpleApp(const ProgramArguments &args);

    void initializeDriveWorks(dwContextHandle_t &context) const;

    std::string statEnumToString(dwSerializerStatTimeDifference stat)
    {
        std::string s;
        switch (stat)
        {
        case DW_SERIALIZER_STAT_DISK_WRITE_TIME:
            s = "DW_SERIALIZER_STAT_DISK_WRITE_TIME";
            break;
        case DW_SERIALIZER_STAT_ENCODE_TIME:
            s = "DW_SERIALIZER_STAT_ENCODE_TIME";
            break;
        case DW_SERIALIZER_STAT_STAGE1_TIME:
            s = "DW_SERIALIZER_STAT_STAGE1_TIME";
            break;
        case DW_SERIALIZER_STAT_STAGE2_TIME:
            s = "DW_SERIALIZER_STAT_STAGE2_TIME";
            break;
        case DW_SERIALIZER_STAT_STAGE3_TIME:
            s = "DW_SERIALIZER_STAT_STAGE3_TIME";
            break;
        case DW_SERIALIZER_STAT_COUNT:
            s = "";
            break;
        default:
            s = "";
            break;
        }
        return s;
    }

    void logStats(uint32_t serializerIdx)
    {
        if (!m_bUseNewCodecStack)
        {
            dwSerializerStats *stats = new dwSerializerStats;
            for (uint32_t i = 0; i < m_totalCameras; ++i)
            {
                dwSensorSerializer_getStats(stats, m_serializer[serializerIdx]);
                const char *cameraName;
                dwRig_getSensorName(&cameraName, i, m_rigConfig);
                std::cout << "Serializer statistics for sensor: " << cameraName << std::endl;
                for (uint32_t stat = 0; stat < DW_SERIALIZER_STAT_COUNT; stat++)
                {
                    std::cout << "Statistics for " << statEnumToString(static_cast<dwSerializerStatTimeDifference>(stat)) << std::endl;
                    std::cout << "minDelta: " << stats->minDeltaUs[stat] << " us" << std::endl;
                    std::cout << "maxDelta: " << stats->maxDeltaUs[stat] << " us" << std::endl;
                    std::cout << "meanDelta: " << stats->meanDelta[stat] << " us" << std::endl;
                    std::cout << "standardDeviationDelta: " << stats->standardDeviationDelta[stat] << std::endl;
                }
            }
            delete stats;
        }
    }

    // Sample framework
    void onProcess() override final {}
    void onRelease() override final
    {
        // Destroy the CUDA generator first: it owns device/pinned buffers and
        // a stream, none of which depend on the DriveWorks context.
        m_eventGen.reset();
        if (m_eventCsv.is_open())
            m_eventCsv.close();
        if (m_streamerRawToCUDA)
        {
            dwImageStreamer_release(m_streamerRawToCUDA);
            m_streamerRawToCUDA = DW_NULL_HANDLE;
        }

        CHECK_DW_ERROR(dwSensor_stop(m_camera[FIRST_CAMERA_IDX]));
        for (uint32_t idx = 0; idx < dwCameraISPType::DW_CAMERA_MAX_ISP_COUNT; idx++)
        {
            if (m_serializer[idx] != nullptr)
            {
                logStats(idx);
                dwSensorSerializer_stop(m_serializer[idx]);
                dwSensorSerializer_release(m_serializer[idx]);
            }

            if (m_serializerNew[idx] != nullptr)
            {
                CHECK_DW_ERROR(dwSensorSerializerNew_stop(m_serializerNew[idx]));
                CHECK_DW_ERROR(dwSensorSerializerNew_release(m_serializerNew[idx]));
            }
        }

        if (m_codecHeader != nullptr)
        {
            CHECK_DW_ERROR(dwCodecHeader_destroy(m_codecHeader));
        }

        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {
            if (m_streamerToCPUGrab[i])
            {
                dwImageStreamer_release(m_streamerToCPUGrab[i]);
            }
            if (m_streamerToGL[i])
            {
                dwImageStreamerGL_release(m_streamerToGL[i]);
            }
            if (m_camera[i])
            {
                dwSAL_releaseSensor(m_camera[i]);
            }
        }

        m_screenshot.reset();
        if (m_rigConfig)
            dwRig_release(m_rigConfig);
        if (m_cameraMaster)
            dwSAL_releaseSensor(m_cameraMaster);
        if (m_sal)
            dwSAL_release(m_sal);
        if (m_renderEngine)
            dwRenderEngine_release(m_renderEngine);
        if (m_renderer)
            dwRenderer_release(m_renderer);
        if (m_viz)
            dwVisualizationRelease(m_viz);
        if (m_streamerEvToGL)
            dwImageStreamerGL_release(m_streamerEvToGL);
        if (m_evImg)
            dwImage_destroy(m_evImg);
        if (m_context)
            dwRelease(m_context);
    }

    void onKeyDown(int key, int scancode, int mods) override
    {
        (void)scancode;
        (void)mods;

        if (key == GLFW_KEY_S)
        {
            m_screenshot->triggerScreenshot();
        }

        if (key == GLFW_KEY_F)
        {
            m_frameGrab = true;
        }

        if (key == GLFW_KEY_P)
        {
            m_startSensor = false;
            m_resetSensor = false;
            if (m_pauseSensor)
            {
                m_isCameraSelected = false;
            }
            m_pauseSensor = !m_pauseSensor;
            std::cout << "Pause sensor toggled " << m_pauseSensor << std::endl;
        }

        if (key == GLFW_KEY_O)
        {
            m_resetSensor = false;
            m_pauseSensor = false;

            if (m_startSensor)
            {
                m_isCameraSelected = false;
            }
            m_startSensor = !m_startSensor;
            std::cout << "Start sensor toggled " << m_startSensor << std::endl;
        }

        if (key == GLFW_KEY_I)
        {
            m_startSensor = false;
            m_pauseSensor = false;

            if (m_resetSensor)
            {
                m_isCameraSelected = false;
            }
            m_resetSensor = !m_resetSensor;
            std::cout << "Reset sensor toggled " << m_resetSensor << std::endl;
        }

        if ((key >= GLFW_KEY_0 && key <= GLFW_KEY_9) || (key >= GLFW_KEY_A && key <= GLFW_KEY_F))
            if (m_pauseSensor || m_startSensor || m_resetSensor)
            {
                if (!m_isCameraSelected)
                {
                    // between KEY_9 and KEY_A there are 2 symbols : and ;, so we skip them
                    m_cameraSelected = key - ((key < GLFW_KEY_A) ? GLFW_KEY_0 : GLFW_KEY_0 + 2);
                    m_isCameraSelected = true;
                }
            }
    }

    ///------------------------------------------------------------------------------
    /// Image stream frame grabber.
    ///     - read from camera
    ///     - get an image with a useful format
    ///     - save image to file
    ///------------------------------------------------------------------------------
    void frameGrab(dwImageHandle_t frameCUDA, uint32_t index)
    {
        dwTime_t timeout = 500000;

        // stream that image to the CPU domain
        CHECK_DW_ERROR(dwImageStreamer_producerSend(frameCUDA, m_streamerToCPUGrab[index]));

        // receive the streamed image as a handle
        dwImageHandle_t frameCPU;
        CHECK_DW_ERROR(dwImageStreamer_consumerReceive(&frameCPU, timeout, m_streamerToCPUGrab[index]));

        // get an image from the frame
        dwImageCPU *imgCPU;
        CHECK_DW_ERROR(dwImage_getCPU(&imgCPU, frameCPU));

        // write the image to a file
        char fname[128];
        dwTime_t timestamp;
        dwImage_getTimestamp(&timestamp, frameCPU);
        if (timestamp == 0)
        {
            timestamp = m_screenshotCount++;
        }
        sprintf(fname, "GMSL_IDX%s_framegrab_%s.png", std::to_string(index).c_str(), std::to_string(timestamp).c_str());
        uint32_t error = lodepng_encode32_file(fname, imgCPU->data[0], imgCPU->prop.width, imgCPU->prop.height);
        std::cout << "Frame Grab saved to " << fname << " " << error << "\n";

        // reset frame grab flag
        // returned the consumed image
        CHECK_DW_ERROR(dwImageStreamer_consumerReturn(&frameCPU, m_streamerToCPUGrab[index]));

        // notify the producer that the work is done
        CHECK_DW_ERROR(dwImageStreamer_producerReturn(nullptr, timeout, m_streamerToCPUGrab[index]));
    }

    /**
     * @brief Specify engines/modules that need to be used to interface camera frames with.
     * @note Camera frames can be used for serialization, which require to setup Encoder
     *          attributes as part of the image pool.
     *
     * @param cameraIdx : Index of camera
     * @param outputType : OutputType supported with Camera
     * @param serializerIdx : Serializer corresponding to the pipeline.
     */
    void setImageAttributes(uint32_t cameraIdx, dwCameraOutputType outputType, uint32_t const serializerIdx)
    {
        std::cout << "Setting Image Pool for CameraOutputType: " << outputType;

        dwImageProperties imgProps{};

        // Gets information about the image properties for a given 'dwCameraImageOutputType'.
        CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&imgProps, outputType, m_camera[cameraIdx]));

        // Append the NvSci allocation attribute of Camera Module.
        CHECK_DW_ERROR(dwSensorCamera_appendAllocationAttributes(&imgProps, outputType, m_camera[cameraIdx]));

        if (m_recordCamera)
        {
            if (m_bUseNewCodecStack && (m_serializerNew[serializerIdx] != DW_NULL_HANDLE))
            {
                CHECK_DW_ERROR(dwSensorSerializerNew_appendAllocationAttributes(&imgProps, m_serializerNew[serializerIdx]));
            }
            else if (m_serializer[serializerIdx] != DW_NULL_HANDLE)
            {
                CHECK_DW_ERROR(dwSensorSerializer_appendAllocationAttributes(&imgProps, m_serializer[serializerIdx]));
            }
            else
            {
                throw std::runtime_error("Invalid serializers. Initialize Serialziers before requesting for serializers Attributes \n");
            }
        }

        CHECK_DW_ERROR(dwSensorCamera_setImageAttributes(&imgProps, outputType, m_camera[cameraIdx]));
    }

    /**
     * @brief Sets and creates a pool of image to be used as output by the camera layer.
     *         - Append Camera Module NvSci Attributes to imageProperties.
     *         - This can be extended to append Other Engine Attributes.
     *         - Creates Images based on input parameters to be used to Camera Module.
     *         - If this is called, the default pool is not allocated.
     *
     * @param cameraIdx : Active Camera Index
     * @param imagePoolIndex : Index for associated Pipeline for which ImagePool is being created
     * @param outputType : Ouptut type of the image.
     */
    void setImagePool(uint32_t cameraIdx, uint32_t imagePoolIndex, dwCameraOutputType outputType, uint32_t const serializerIdx)
    {
        dwImagePool pool{};

        std::cout << "Setting Image Pool for CameraOutputType: " << outputType;

        for (uint32_t i = 0; i < m_fifoSize; i++)
        {
            dwImageProperties imgProps{};

            // Gets information about the image properties for a given 'dwCameraImageOutputType'.
            CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&imgProps, outputType, m_camera[cameraIdx]));

            // Append the NvSci allocation attribute of Camera Module.
            CHECK_DW_ERROR(dwSensorCamera_appendAllocationAttributes(&imgProps, outputType, m_camera[cameraIdx]));

            if (m_recordCamera)
            {
                if (m_bUseNewCodecStack && (m_serializerNew[serializerIdx] != DW_NULL_HANDLE))
                {
                    CHECK_DW_ERROR(dwSensorSerializerNew_appendAllocationAttributes(&imgProps, m_serializerNew[serializerIdx]));
                }
                else if (m_serializer[serializerIdx] != DW_NULL_HANDLE)
                {
                    CHECK_DW_ERROR(dwSensorSerializer_appendAllocationAttributes(&imgProps, m_serializer[serializerIdx]));
                }
                else
                {
                    throw std::runtime_error("Invalid serializers. Initialize serializers before requesting for serializer Attributes \n");
                }
            }

            CHECK_DW_ERROR(dwImage_create(&m_imagesForImagePool[cameraIdx][imagePoolIndex][i], imgProps, m_context));
        }

        // pass the frames to the camera.
        pool.imageCount = m_fifoSize;
        pool.images = &m_imagesForImagePool[cameraIdx][imagePoolIndex][0];

        CHECK_DW_ERROR(dwSensorCamera_setImagePool(pool, m_camera[cameraIdx]));
    }

    /**
     * @brief Helper method to set and allocate and exchange buffers and synchronization objects across Camera Sensor and external modules/applications.
     */
    void setNvSciAttributesHelper()
    {
        std::cout << "Main: Setting Image Pool." << std::endl;

        // NvSciBuf
        for (uint32_t idx = 0; idx < m_totalCameras; ++idx)
        {
            if (m_useRaw)
            {
                // Create images to set the ImagePool for ICP Pipeline.
                setImagePool(idx, ICP_PIPELINE_IDX, dwCameraOutputType::DW_CAMERA_OUTPUT_NATIVE_RAW, 0);
            }

            if (m_useProcessed)
            {
                // Create images to set the ImagePool for ISP0 Pipeline.
                setImagePool(idx, ISP_PIPELINE_IDX, dwCameraOutputType::DW_CAMERA_OUTPUT_NATIVE_PROCESSED, static_cast<uint32_t>(dwCameraISPType::DW_CAMERA_ISP0));
            }

            if (m_useProcessed1)
            {
                // Create images to set the ImagePool for ISP1 Pipeline.
                setImagePool(idx, ISP_PIPELINE_IDX, dwCameraOutputType::DW_CAMERA_OUTPUT_NATIVE_PROCESSED1, static_cast<uint32_t>(dwCameraISPType::DW_CAMERA_ISP1));
            }

            if (m_useProcessed2)
            {
                // Create images to set the ImagePool for ISP2 Pipeline.
                setImagePool(idx, ISP_PIPELINE_IDX, dwCameraOutputType::DW_CAMERA_OUTPUT_NATIVE_PROCESSED2, static_cast<uint32_t>(dwCameraISPType::DW_CAMERA_ISP2));
            }
        }
    }

    dwStatus initializeSerializer(dwSensorParams const &paramsClient, std::string fileName, uint32_t pipeLineIdx = 0)
    {
        if (m_bUseNewCodecStack)
        {
            return initializeSerializerNew(paramsClient, fileName, pipeLineIdx);
        }

        dwSerializerParams serializerParams;
        std::string newParams(paramsClient.parameters);

        // sample_camera recorder does not support async-record
        std::string asyncRecordStr(",async-record=1");
        size_t pos = newParams.find(asyncRecordStr);

        if (pos != std::string::npos)
        {
            newParams.erase(newParams.find(asyncRecordStr), asyncRecordStr.size());
        }

        newParams += std::string(",type=disk,file=") + fileName;

        if (pipeLineIdx == dwCameraISPType::DW_CAMERA_ISP1)
        {
            newParams += ",isp-type=1";
        }

        serializerParams.parameters = newParams.c_str();
        serializerParams.onData = nullptr;

        CHECK_DW_ERROR(dwSensorSerializer_initialize(&(m_serializer[pipeLineIdx]), &serializerParams, m_camera[FIRST_CAMERA_IDX]));
        CHECK_DW_ERROR(dwSensorSerializer_start(m_serializer[pipeLineIdx]));

        return DW_SUCCESS;
    }

    dwStatus initializeSerializerNew(dwSensorParams const &paramsClient, std::string fileName, uint32_t pipeLineIdx = 0)
    {
        dwSerializerParams serializerParams;
        std::string newParams(paramsClient.parameters);

        // new codec stack only supports async-record
        newParams += std::string(",async-record=1,type=disk,file=") + fileName;

        if (pipeLineIdx == dwCameraISPType::DW_CAMERA_ISP1)
        {
            newParams += ",isp-type=1";
            newParams += ",image-type=";
            auto imageType = std::to_string(DW_CAMERA_OUTPUT_NATIVE_PROCESSED1);
            newParams += imageType.c_str();
            newParams += ",";
        }

        if (m_recordFormat == RECORD_FILE_FORMAT_LRAW || m_recordFormat == RECORD_FILE_FORMAT_RAW)
        {
            throw std::runtime_error("lraw/raw video format encoding is not supported on new codec stack. Please switch to pod \n");
        }

        if (m_recordFormat == RECORD_FILE_FORMAT_H264 || m_recordFormat == RECORD_FILE_FORMAT_H265)
        {
            throw std::runtime_error("h264/h265 video format encoding is not supported on new codec stack. Plese switch to mp4 \n");
        }

        serializerParams.parameters = newParams.c_str();
        serializerParams.onData = nullptr;

        dwCameraProperties cameraProperties{};
        CHECK_DW_ERROR(dwSensorCamera_getSensorProperties(&cameraProperties, m_camera[FIRST_CAMERA_IDX]));

        dwImageProperties imageProperties;
        if (pipeLineIdx == dwCameraISPType::DW_CAMERA_ISP0)
        {
            CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&imageProperties, DW_CAMERA_OUTPUT_NATIVE_PROCESSED, m_camera[FIRST_CAMERA_IDX]));
        }
        else if (pipeLineIdx == dwCameraISPType::DW_CAMERA_ISP1)
        {
            CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&imageProperties, DW_CAMERA_OUTPUT_NATIVE_PROCESSED1, m_camera[FIRST_CAMERA_IDX]));
        }
        else if (pipeLineIdx == dwCameraISPType::DW_CAMERA_ISP2)
        {
            CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&imageProperties, DW_CAMERA_OUTPUT_NATIVE_PROCESSED2, m_camera[FIRST_CAMERA_IDX]));
        }

        // Create encoder
        dwEncoderConfig encoderConfig{};
        encoderConfig.rateControl.quality = 20;
        encoderConfig.rateControl.gopSize = 5;
        encoderConfig.rateControl.averageBitRate = 200000000;
        encoderConfig.rateControl.maxBitRate = 200000000 * 4;
        encoderConfig.rateControl.mode = DW_ENCODER_RATE_CONTROL_MODE_CONSTQP;

        dwCodecConfigVideo configVideo{};
        configVideo.width = imageProperties.width;
        configVideo.height = imageProperties.height;
        configVideo.frameRate = cameraProperties.framerate;
        configVideo.bitDepth = cameraProperties.imageBitDepth;
        configVideo.format = DW_CODEC_VIDEO_FORMAT_YUV420;
        configVideo.rateControl = encoderConfig.rateControl;

        dwCodecMimeTypeConst_t codecType = DW_CODEC_MIME_TYPE_VIDEO_H264;
        if (m_recordFormat == RECORD_FILE_FORMAT_H265)
        {
            codecType = DW_CODEC_MIME_TYPE_VIDEO_H265;
        }

        std::strncpy(configVideo.codecMimeType, codecType.mime, DW_MAX_CODEC_MIME_TYPE_LENGTH);

        CHECK_DW_ERROR(dwCodecHeader_createNew(&m_codecHeader, codecType.mime, &configVideo, nullptr, m_context));

        CHECK_DW_ERROR(dwSensorSerializerNew_initialize(&(m_serializerNew[pipeLineIdx]), m_codecHeader, &serializerParams, m_context));

        CHECK_DW_ERROR(dwSensorSerializerNew_start(m_serializerNew[pipeLineIdx]));

        return DW_SUCCESS;
    }

    void getRecordFileFormat(std::string format)
    {
        if (format.compare("h264") == 0)
        {
            m_recordFormat = RECORD_FILE_FORMAT_H264;
        }
        else if (format.compare("h265") == 0)
        {
            m_recordFormat = RECORD_FILE_FORMAT_H265;
        }
        else if (format.compare("raw") == 0)
        {
            m_recordFormat = RECORD_FILE_FORMAT_RAW;
        }
        else if (format.compare("lraw") == 0)
        {
            m_recordFormat = RECORD_FILE_FORMAT_LRAW;
        }
        else if (format.compare("mp4") == 0)
        {
            m_recordFormat = RECORD_FILE_FORMAT_MP4;
        }
        else if (format.compare("pod") == 0)
        {
            m_recordFormat = RECORD_FILE_FORMAT_POD;
        }
        else
        {
            m_recordFormat = RECORD_FILE_FORMAT_UNSUPPORTED;
        }
    }

    bool onInitialize() override
    {
        // -----------------------------------------
        // Initialize DriveWorks context and SAL
        // -----------------------------------------

        // initialize logger to print verbose message on console in color
        dwLogger_initialize(getConsoleLoggerCallback(true));
        dwLogger_setLogLevel(DW_LOG_VERBOSE);

        initializeDriveWorks(m_context);
        CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_context));

        // if rig is selected
        m_totalCameras = 0;

        // the complete configuration is automatically loaded by dwRig module
        CHECK_DW_ERROR(dwRig_initializeFromFile(&m_rigConfig, m_context,
                                                getArgument("rig").c_str()));

        uint32_t cnt = 0;
        CHECK_DW_ERROR(dwRig_getSensorCountOfType(&cnt, DW_SENSOR_CAMERA, m_rigConfig));

        // initialize error handling flag
        isResetLinkOnError = (std::stoi(getArgument("reset-on-error")) > 0);
        bool isEnableUserCallback = (std::stoi(getArgument("enable-user-event-callbacks")) > 0);
        m_isProcessed1WriteEnabled = (std::stoi(getArgument("enable-processed1-write")) > 0);

        //------------------------------------------------------------------------------
        // initializes cameras
        // -----------------------------------------
        dwSensorParams paramsClient[MAX_CAMS] = {};
        for (uint32_t i = 0; i < cnt; i++)
        {
            uint32_t cameraSensorIdx = 0;
            CHECK_DW_ERROR(dwRig_findSensorByTypeIndex(&cameraSensorIdx, DW_SENSOR_CAMERA, i, m_rigConfig));

            // get rig parsed protocol from dwRig
            const char *protocol = nullptr;
            CHECK_DW_ERROR(dwRig_getSensorProtocol(&protocol, cameraSensorIdx, m_rigConfig));
            const char *params = nullptr;
            CHECK_DW_ERROR(dwRig_getSensorParameterUpdatedPath(&params, cameraSensorIdx, m_rigConfig));
            paramsClient[i].protocol = protocol;
            paramsClient[i].parameters = params;

            // if image from isp0 is yuv420-uint8-bl or yuv420-uint16-bl then disable the gl render.
            m_enableRender[i] = (strstr(params, PATTERNYUV420BL1) == NULL && strstr(params, PATTERNYUV420BL2) == NULL) ? true : false;

            std::cout << "onInitialize: creating camera.gmsl with params: " << params << std::endl;
            CHECK_DW_ERROR(dwSAL_createSensor(&m_camera[m_totalCameras], paramsClient[i], m_sal));

            // register error handler
            if (isEnableUserCallback)
            {
                CHECK_DW_ERROR(dwSensorCamera_setEventCallback(cameraDevBlkEventHandling, cameraPipelineEventHandling, m_camera[m_totalCameras]));
            }

            m_totalCameras++;

            m_useRaw = std::string::npos != std::string(params).find("raw");
            m_useProcessed = std::string::npos != std::string(params).find("processed");
            m_useProcessed1 = std::string::npos != std::string(params).find("processed1");
            m_useProcessed2 = std::string::npos != std::string(params).find("processed2");

            m_bUseNewCodecStack = false;
            std::string newCodecStackStr("newcodecstack=1");
            size_t pos = std::string(params).find(newCodecStackStr);

            if (pos != std::string::npos)
            {
                m_bUseNewCodecStack = true;
            }

            if (!m_useProcessed && !m_useRaw) // if neither output format specified, assume processed output
                m_useProcessed = true;
            if (!m_useProcessed)
                std::cout << "Processed not selected as master output format parameter. No images will be previewed." << std::endl;

            auto posFifoSize = std::string(params).find("fifo-size=");
            if (posFifoSize != std::string::npos)
            {
                static constexpr uint32_t COMMA_INCLUDED_STR_MAX_SIZE = 5;
                auto fifoSizeStr = std::string(params).substr(posFifoSize + std::string("fifo-size=").size(), COMMA_INCLUDED_STR_MAX_SIZE);
                auto fifoSizeStrDataPos = fifoSizeStr.find(",");
                if ((fifoSizeStrDataPos != std::string::npos) && (fifoSizeStr.size() != 0))
                {
                    m_fifoSize = std::stoi(fifoSizeStr.substr(0, fifoSizeStrDataPos));
                }
            }

            if (m_fifoSize == 0)
            {
                // Minimum Recommended FifoSize
                m_fifoSize = 4;
            }
        }

        CHECK_DW_ERROR(dwSAL_start(m_sal));

        

        //------------------------------------------------------------------------------
        // initializes streamers
        // -----------------------------------------
        if (m_useProcessed)
        {
            dwImageProperties imageProperties{};
            for (uint32_t i = 0; i < m_totalCameras; ++i)
            {
                std::cout << "onInitialize: getting image props " << i << std::endl;
                CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&imageProperties, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, m_camera[i]));

                std::cout << "onInitialize: initilizing stream: " << i << std::endl;
                CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_streamerToGL[i], &imageProperties, DW_IMAGE_GL, m_context));

                CHECK_DW_ERROR(dwImageStreamer_initialize(&m_streamerToCPUGrab[i], &imageProperties, DW_IMAGE_CPU, m_context));
            }
        }

        //------------------------------------------------------------------------------
        // initializes evsim CUDA event generator
        // -----------------------------------------
        if (std::stoi(getArgument("enable-events")) > 0)
        {
            m_eventCamera = static_cast<uint32_t>(std::stoi(getArgument("event-camera")));
            m_eventUseRaw = (getArgument("event-input") == "raw");
            m_eventRawShift = std::stoi(getArgument("event-raw-shift"));
            if (m_eventUseRaw && !m_useRaw)
            {
                logWarn("event-input=raw but the rig does not request a raw output "
                        "(add 'raw' to output-requested, or use --event-input=rgba); events disabled.\n");
            }
            else if (!m_eventUseRaw && !m_useProcessed)
            {
                logWarn("event-input=rgba needs a processed (RGBA) output; events disabled.\n");
            }
            else if (m_eventCamera >= m_totalCameras)
            {
                logWarn("event-camera %u out of range (%u cameras); events disabled.\n",
                        m_eventCamera, m_totalCameras);
            }
            else
            {
                dwImageProperties evProps{};
                // const dwCameraOutputType evType = m_eventUseRaw ? DW_CAMERA_OUTPUT_NATIVE_RAW
                //                                                 : DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8;
                const dwCameraOutputType evType = m_eventUseRaw ?  DW_CAMERA_OUTPUT_CUDA_RAW_UINT16 
                                                                : DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8;
                CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&evProps, evType, m_camera[m_eventCamera]));
                m_eventW = evProps.width;
                m_eventH = evProps.height;

                log("onInitialize: event input %s: type %d format %d %ux%u\n",
                    m_eventUseRaw ? "raw" : "rgba",
                    static_cast<int>(evProps.type), static_cast<int>(evProps.format),
                    evProps.width, evProps.height);

                if (m_eventUseRaw)
                {
                    // NvSci raw image -> CUDA image (pitch-linear, device-accessible).
                    CHECK_DW_ERROR(dwImageStreamer_initialize(&m_streamerRawToCUDA, &evProps, DW_IMAGE_CUDA, m_context));
                }

                evsim::Config cfg;
                cfg.sensorType = getArgument("event-sensor");
                cfg.fastDeterministicMode = std::stoi(getArgument("event-deterministic")) > 0;
                cfg.contrastThresholdOn = std::stod(getArgument("event-threshold-on"));
                cfg.contrastThresholdOff = std::stod(getArgument("event-threshold-off"));

                const int maxEvents = std::stoi(getArgument("event-max"));
                m_eventGen.reset(new evsim::EventGeneratorCUDA(cfg, maxEvents));
                // Explicit initialize(): the generator does not handle resolution
                // changes mid-stream, so size it from the real camera output.
                m_eventW = evProps.width;
                m_eventH = evProps.height;
                m_eventGen->initialize(static_cast<int>(evProps.width), static_cast<int>(evProps.height));

                m_evOn.reserve(m_eventDrawCap);
                m_evOff.reserve(m_eventDrawCap);

                m_lumaBuf.assign(static_cast<size_t>(evProps.width) * evProps.height, 0);

                const std::string &csvPath = getArgument("event-csv");
                if (!csvPath.empty())
                {
                    m_eventCsv.open(csvPath);
                    m_eventCsv << "x,y,timestamp_s,polarity\n";
                }

                log("onInitialize: event generator on camera %u (%ux%u, %s mode)\n",
                    m_eventCamera, evProps.width, evProps.height,
                    cfg.fastDeterministicMode ? "deterministic" : "stochastic");
            }
        }

        // -----------------------------
        // Initialize Renderer
        // -----------------------------
        {
            CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));
            m_colorPerPort[0] = {1, 0, 0, 1};
            m_colorPerPort[1] = {0, 1, 0, 1};
            m_colorPerPort[2] = {0, 0, 1, 1};
            m_colorPerPort[3] = {0, 0, 0, 1};

            log("onInitialize: Total cameras %d\n", m_totalCameras);

            CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params, getWindowWidth(), getWindowHeight()));
            params.defaultTile.lineWidth = 2.0f;
            params.defaultTile.font = DW_RENDER_ENGINE_FONT_VERDANA_24;
            params.maxBufferCount = 1;

            float32_t windowSize[2] = {static_cast<float32_t>(getWindowWidth()), static_cast<float32_t>(getWindowHeight())};
            params.bounds = {0, 0, 0, 0};

            /***
            uint32_t tilesPerRow = 1;
            params.bounds.width = windowSize[0];
            params.bounds.height = windowSize[1];
            switch (m_totalCameras)
            {
            case 1:
                tilesPerRow = 1;
                break;
            case 2:
                params.bounds.height = (windowSize[1] / 2);
                params.bounds.y = (windowSize[1] / 2);
                tilesPerRow = 2;
                break;
            case 3:
                tilesPerRow = 2;
                break;
            case 4:
                tilesPerRow = 2;
                break;
            default:
                tilesPerRow = 4;
                break;
            }

            CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_viz));

            dwRenderEngineTileState paramList[MAX_CAMS];
            for (uint32_t i = 0; i < m_totalCameras; ++i)
            {
                dwRenderEngine_initTileState(&paramList[i]);
                paramList[i].modelViewMatrix = DW_IDENTITY_MATRIX4F;
                paramList[i].font = DW_RENDER_ENGINE_FONT_VERDANA_24;
            }

            dwRenderEngine_addTilesByCount(m_tileVideo, m_totalCameras, tilesPerRow, paramList, m_renderEngine);
            */
            m_eventDisplay = (std::stoi(getArgument("enable-events")) > 0) &&
                             (std::stoi(getArgument("event-display")) > 0);
            const uint32_t tileCount = m_totalCameras + (m_eventDisplay ? 1 : 0);

            uint32_t tilesPerRow = 1;
            params.bounds.width = windowSize[0];
            params.bounds.height = windowSize[1];
            switch (tileCount)
            {
            case 1:
                tilesPerRow = 1;
                break;
            case 2:
                params.bounds.height = (windowSize[1] / 2);
                params.bounds.y = (windowSize[1] / 2);
                tilesPerRow = 2;
                break;
            case 3:
            case 4:
                tilesPerRow = 2;
                break;
            default:
                tilesPerRow = 4;
                break;
            }

            CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_viz));

            dwRenderEngineTileState paramList[MAX_CAMS + 1];
            for (uint32_t i = 0; i < tileCount; ++i)
            {
                dwRenderEngine_initTileState(&paramList[i]);
                paramList[i].modelViewMatrix = DW_IDENTITY_MATRIX4F;
                paramList[i].font = DW_RENDER_ENGINE_FONT_VERDANA_24;
            }
            // dark background for the event tile
            if (m_eventDisplay)
                paramList[m_totalCameras].backgroundColor = {0.05f, 0.05f, 0.05f, 1.0f};

            dwRenderEngine_addTilesByCount(m_tileVideo, tileCount, tilesPerRow, paramList, m_renderEngine);

            if (m_eventDisplay)
            {

                if (m_eventW == 0 || m_eventH == 0)
                    throw std::runtime_error("event display: camera reported 0x0 image");

                m_evOnCnt.assign(size_t(m_eventW) * m_eventH, 0);
                m_evOffCnt.assign(size_t(m_eventW) * m_eventH, 0);

                dwImageProperties p{};
                p.type = DW_IMAGE_CPU;
                p.format = DW_IMAGE_FORMAT_RGBA_UINT8;
                p.memoryLayout = DW_IMAGE_MEMORY_TYPE_PITCH;
                p.width = m_eventW;
                p.height = m_eventH;

                log("event display: creating %ux%u CPU RGBA image + GL streamer\n", p.width, p.height);

                CHECK_DW_ERROR(dwImage_create(&m_evImg, p, m_context));
                CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_streamerEvToGL, &p, DW_IMAGE_GL, m_context));
            }
        }

        //------------------------------------------------------------------------------
        // initializes recorder serializer
        // -----------------------------------------
        {
            const std::string &videoFile = getArgument("write-file");
            m_recordCamera = !videoFile.empty();

            if (m_recordCamera)
            {
                // Determine format from file extension
                std::string format;
                size_t found = videoFile.find_last_of(".");

                if (found != std::string::npos)
                {
                    format = videoFile.substr(found + 1);
                }

                if (format.compare("h264") != 0 && format.compare("h265") != 0 && format.compare("mp4") != 0 && format.compare("raw") != 0 && format.compare("lraw") != 0)
                {
                    logError("Unsupported output format. Must be one of h264, h265, mp4 or raw/lraw\n");
                    return false;
                }

                if (format.compare("raw") == 0 || format.compare("lraw") == 0)
                    m_recordRaw = true;

                getRecordFileFormat(format);

                // Initialize Base Serializer (raw/ISP0)
                dwStatus status = initializeSerializer(paramsClient[FIRST_CAMERA_IDX], std::string(getArgument("write-file")), 0);
                if (status != DW_SUCCESS)
                {
                    logError("Initializing serializer for raw/processed pipleline failed\n");
                    return false;
                }

                if (m_useProcessed1 && m_isProcessed1WriteEnabled)
                {
                    if (format.compare("h264") == 0 || format.compare("h265") == 0 || format.compare("mp4") == 0)
                    {
                        std::string fileName = videoFile.substr(0, found) + std::string("-processed1.") + format;
                        status = initializeSerializer(paramsClient[FIRST_CAMERA_IDX], fileName, dwCameraISPType::DW_CAMERA_ISP1);
                        if (status != DW_SUCCESS)
                        {
                            logError("Initializing serializer for processed1 pipleline failed\n");
                            return false;
                        }
                    }
                }
            }
        }

        if (m_recordCamera)
        {
            setNvSciAttributesHelper();
        }

        m_screenshot.reset(new ScreenshotHelper(m_context, m_sal, getWindowWidth(), getWindowHeight(), "CameraGMSLCustom"));

        std::cout << "Main: Starting master." << std::endl;
        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {
            CHECK_DW_ERROR(dwSensor_start(m_camera[i]));
        }

        return true;
    }

    // Common bookkeeping for a finished packet: counters, CSV, periodic log.
    void publishEvents(const evsim::EventPacket &packet, uint32_t cameraIndex)
    {
        m_eventLastPacket = packet.size();
        m_eventTotal += packet.size();
        m_eventFrames++;

        // if (m_eventDisplay)
        // {
        //     m_evOn.clear();
        //     m_evOff.clear();
        //     for (const evsim::Event &e : packet.events)
        //     {
        //         dwVector2f p{static_cast<float32_t>(e.x), static_cast<float32_t>(e.y)};
        //         if (e.polarity > 0)
        //         {
        //             if (m_evOn.size() < m_eventDrawCap)
        //                 m_evOn.push_back(p);
        //         }
        //         else
        //         {
        //             if (m_evOff.size() < m_eventDrawCap)
        //                 m_evOff.push_back(p);
        //         }
        //     }
        // }

        if (m_eventDisplay)
        {
            for (const evsim::Event &e : packet.events)
            {
                if (e.x >= m_eventW || e.y >= m_eventH)
                    continue;
                uint16_t &c = (e.polarity > 0 ? m_evOnCnt : m_evOffCnt)[size_t(e.y) * m_eventW + e.x];
                if (c < 0xFFFF)
                    ++c;
            }
        }

        if (m_eventCsv.is_open())
        {
            for (const evsim::Event &e : packet.events)
            {
                m_eventCsv << e.x << ',' << e.y << ',' << e.timestamp << ','
                           << static_cast<int>(e.polarity) << '\n';
            }
        }

        if (m_eventFrames % 30 == 0)
        {
            log("events: cam %u frame %lu -> %zu events (total %lu)\n",
                cameraIndex, static_cast<unsigned long>(packet.frameNumber),
                m_eventLastPacket, static_cast<unsigned long>(m_eventTotal));
        }
    }

    ///------------------------------------------------------------------------------
    /// Raw path: feed the NATIVE_RAW plane directly to the generator on the GPU.
    ///     - get the raw image of the frame
    ///     - stream it to a CUDA image (pitch-linear)
    ///     - generateFromDevice(): repack (+ optional bit shift) and simulate
    ///------------------------------------------------------------------------------
    void processEventsRaw(dwCameraFrameHandle_t frame, uint32_t cameraIndex)
    {
        dwImageHandle_t raw = DW_NULL_HANDLE;
        // CHECK_DW_ERROR(dwSensorCamera_getImage(&raw, DW_CAMERA_OUTPUT_NATIVE_RAW, frame));
        CHECK_DW_ERROR(dwSensorCamera_getImage(&raw,  DW_CAMERA_OUTPUT_CUDA_RAW_UINT16 , frame));

        dwTime_t timestampUs = 0;
        CHECK_DW_ERROR(dwImage_getTimestamp(&timestampUs, raw));

        const dwTime_t timeout = 500000;
        CHECK_DW_ERROR(dwImageStreamer_producerSend(raw, m_streamerRawToCUDA));

        dwImageHandle_t frameCUDA = DW_NULL_HANDLE;
        CHECK_DW_ERROR(dwImageStreamer_consumerReceive(&frameCUDA, timeout, m_streamerRawToCUDA));

        dwImageCUDA *imgCUDA = nullptr;
        CHECK_DW_ERROR(dwImage_getCUDA(&imgCUDA, frameCUDA));
        
        log("processEventrsRaw : start of the event packet generation\n");

        evsim::EventPacket packet;
        try
        {
            // generateFromDevice() is synchronous, so the streamer image can be
            // returned right after it.
            // packet = m_eventGen->generateFromDevice(
            //     reinterpret_cast<const uint16_t *>(imgCUDA->dptr[0]),
            //     static_cast<int>(imgCUDA->prop.width),
            //     static_cast<int>(imgCUDA->prop.height),
            //     static_cast<size_t>(imgCUDA->pitch[0]),
            //     static_cast<double>(timestampUs) * 1e-6, // us -> s
            //     m_eventRawShift);
            packet = m_eventGen->generate(
                reinterpret_cast<const uint16_t *>(imgCUDA->dptr[0]),
                static_cast<int>(imgCUDA->prop.width),
                static_cast<int>(imgCUDA->prop.height),
                static_cast<size_t>(imgCUDA->pitch[0]),
                static_cast<double>(timestampUs) * 1e-6 // us -> s
            );
            log("processEvents : finishing event generation task per frame \n");
        
        }
        
        catch (...)
        {
            dwImageStreamer_consumerReturn(&frameCUDA, m_streamerRawToCUDA);
            dwImageStreamer_producerReturn(nullptr, timeout, m_streamerRawToCUDA);
            throw;
        }

        CHECK_DW_ERROR(dwImageStreamer_consumerReturn(&frameCUDA, m_streamerRawToCUDA));
        CHECK_DW_ERROR(dwImageStreamer_producerReturn(nullptr, timeout, m_streamerRawToCUDA));

        publishEvents(packet, cameraIndex);
    }

    // Draw a point list in chunks so each call fits the render engine's internal buffer.
    void drawPoints(const std::vector<dwVector2f> &pts, dwRenderEngineColorRGBA color)
    {
        if (pts.empty())
            return;

        dwRenderEngine_setColor(color, m_renderEngine);

        constexpr size_t CHUNK = 8192; // 64 KB per call
        for (size_t start = 0; start < pts.size(); start += CHUNK)
        {
            const uint32_t n = static_cast<uint32_t>(std::min(CHUNK, pts.size() - start));
            CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_2D,
                                                 pts.data() + start, sizeof(dwVector2f), 0, n,
                                                 m_renderEngine));
        }
    }

    // void renderEventTile()
    // {
    //     if (!m_eventDisplay || m_eventW == 0 || m_eventH == 0)
    //         return;

    //     CHECK_DW_ERROR(dwRenderEngine_setTile(m_tileVideo[m_totalCameras], m_renderEngine));
    //     CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));

    //     // Same coordinate system as the camera image tiles: pixel coords, origin top-left.
    //     dwVector2f range{static_cast<float32_t>(m_eventW), static_cast<float32_t>(m_eventH)};
    //     CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));

    //     CHECK_DW_ERROR(dwRenderEngine_setPointSize(2.0f, m_renderEngine));

    //     if (!m_evOn.empty())
    //     {
    //         // dwRenderEngine_setColor({1.0f, 0.2f, 0.2f, 1.0f}, m_renderEngine); // ON  = red
    //         // CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_2D,
    //         //                                      m_evOn.data(), sizeof(dwVector2f), 0,
    //         //                                      static_cast<uint32_t>(m_evOn.size()), m_renderEngine));
    //         drawPoints(m_evOn, {1.0f, 0.2f, 0.2f, 1.0f}); // ON  = red
    //     }
    //     if (!m_evOff.empty())
    //     {
    //         // dwRenderEngine_setColor({0.2f, 0.5f, 1.0f, 1.0f}, m_renderEngine); // OFF = blue
    //         // CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_2D,
    //         //                                      m_evOff.data(), sizeof(dwVector2f), 0,
    //         //                                      static_cast<uint32_t>(m_evOff.size()), m_renderEngine));
    //         drawPoints(m_evOff, {0.2f, 0.5f, 1.0f, 1.0f}); // OFF = blue
    //     }

    //     // HUD text
    //     dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine);
    //     CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine));
    //     std::string hud = "Events: " + std::to_string(m_eventLastPacket) +
    //                       "  ON:" + std::to_string(m_evOn.size()) +
    //                       "  OFF:" + std::to_string(m_evOff.size());
    //     CHECK_DW_ERROR(dwRenderEngine_renderText2D(hud.c_str(), {10, 25}, m_renderEngine));
    // }

    void renderEventTile()
    {
        if (!m_eventDisplay || m_eventW == 0 || m_eventH == 0)
            return;

        CHECK_DW_ERROR(dwRenderEngine_setTile(m_tileVideo[m_totalCameras], m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));

        if (++m_evFrameCount >= m_evFramesPerWin)
        {
            // Fill the CPU image: ON = red, OFF = blue, dark background.
            dwImageCPU *cpu = nullptr;
            CHECK_DW_ERROR(dwImage_getCPU(&cpu, m_evImg));
            for (uint32_t y = 0; y < m_eventH; ++y)
            {
                uint8_t *dst = cpu->data[0] + size_t(y) * cpu->pitch[0];
                const uint16_t *on = &m_evOnCnt[size_t(y) * m_eventW];
                const uint16_t *off = &m_evOffCnt[size_t(y) * m_eventW];
                for (uint32_t x = 0; x < m_eventW; ++x, dst += 4)
                {
                    dst[0] = uint8_t(std::min(255.0f, on[x] * m_evGain));  // R
                    dst[1] = 0;                                            // G
                    dst[2] = uint8_t(std::min(255.0f, off[x] * m_evGain)); // B
                    dst[3] = 255;                                          // A
                }
            }
            std::fill(m_evOnCnt.begin(), m_evOnCnt.end(), 0);
            std::fill(m_evOffCnt.begin(), m_evOffCnt.end(), 0);
            m_evFrameCount = 0;
        }

        // Draw the CPU image through the GL streamer (same handshake as camera tiles).
        CHECK_DW_ERROR(dwImageStreamerGL_producerSend(m_evImg, m_streamerEvToGL));
        dwImageHandle_t frameGL = DW_NULL_HANDLE;
        CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&frameGL, 33000, m_streamerEvToGL));
        dwImageGL *gl = nullptr;
        CHECK_DW_ERROR(dwImage_getGL(&gl, frameGL));

        dwVector2f range{float32_t(m_eventW), float32_t(m_eventH)};
        CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderImage2D(gl, {0, 0, range.x, range.y}, m_renderEngine));

        CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&frameGL, m_streamerEvToGL));
        CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 32000, m_streamerEvToGL));

        // HUD text (unchanged, but note ON/OFF counts are gone; use m_eventLastPacket)
        dwRenderEngine_setColor({1, 1, 1, 1}, m_renderEngine);
        CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine));
        std::string hud = "Events: " + std::to_string(m_eventLastPacket);
        CHECK_DW_ERROR(dwRenderEngine_renderText2D(hud.c_str(), {10, 25}, m_renderEngine));
    }

    ///------------------------------------------------------------------------------
    /// RGBA fallback path: feed one camera frame to the CUDA event generator.
    ///     - get the CUDA RGBA8 image of the frame
    ///     - stream it to the CPU (same path as frameGrab)
    ///     - convert to a luma plane (uint16, 0..255)
    ///     - run EventGeneratorCUDA::generate()
    ///------------------------------------------------------------------------------

    void processEvents(dwCameraFrameHandle_t frame, uint32_t cameraIndex)
    {
        printf("processEvents : start of event processing \n");
        printf("processEvents : cameraIndex - %d, eventCamera - %d \n", cameraIndex, m_eventCamera);
        int eventGen_check = 0;
        if (m_eventGen)
            eventGen_check = 1;
        printf("processEvents : eventgen check flag - %d \n", eventGen_check);
        if (!m_eventGen || cameraIndex != m_eventCamera)
            return;

        printf("processEvents : eventgen flag checked \n");

        if (m_eventUseRaw)
        {
            processEventsRaw(frame, cameraIndex);
            return;
        }

        dwImageHandle_t img = DW_NULL_HANDLE;
        CHECK_DW_ERROR(dwSensorCamera_getImage(&img, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame));

        dwTime_t timestampUs = 0;
        CHECK_DW_ERROR(dwImage_getTimestamp(&timestampUs, img));

        const dwTime_t timeout = 500000;
        CHECK_DW_ERROR(dwImageStreamer_producerSend(img, m_streamerToCPUGrab[cameraIndex]));

        dwImageHandle_t frameCPU = DW_NULL_HANDLE;
        CHECK_DW_ERROR(dwImageStreamer_consumerReceive(&frameCPU, timeout, m_streamerToCPUGrab[cameraIndex]));

        dwImageCPU *imgCPU = nullptr;
        CHECK_DW_ERROR(dwImage_getCPU(&imgCPU, frameCPU));

        const uint32_t w = imgCPU->prop.width;
        const uint32_t h = imgCPU->prop.height;
        if (m_lumaBuf.size() != static_cast<size_t>(w) * h)
        {
            // Never reallocate silently: generator was sized in onInitialize().
            CHECK_DW_ERROR(dwImageStreamer_consumerReturn(&frameCPU, m_streamerToCPUGrab[cameraIndex]));
            CHECK_DW_ERROR(dwImageStreamer_producerReturn(nullptr, timeout, m_streamerToCPUGrab[cameraIndex]));
            throw std::runtime_error("processEvents: frame size changed after initialization");
        }

        // RGBA8 -> Rec.601 luma, integer approximation, stored as uint16 in 0..255.
        for (uint32_t y = 0; y < h; ++y)
        {
            const uint8_t *src = imgCPU->data[0] + static_cast<size_t>(y) * imgCPU->pitch[0];
            uint16_t *dst = m_lumaBuf.data() + static_cast<size_t>(y) * w;
            for (uint32_t x = 0; x < w; ++x, src += 4)
            {
                dst[x] = static_cast<uint16_t>((77u * src[0] + 150u * src[1] + 29u * src[2]) >> 8);
            }
        }

        // Release the streamer BEFORE the GPU work so the camera pipeline is not
        // held up by generate().
        CHECK_DW_ERROR(dwImageStreamer_consumerReturn(&frameCPU, m_streamerToCPUGrab[cameraIndex]));
        CHECK_DW_ERROR(dwImageStreamer_producerReturn(nullptr, timeout, m_streamerToCPUGrab[cameraIndex]));

        // DriveWorks timestamps are microseconds; the generator expects seconds.
        const double timestampSec = static_cast<double>(timestampUs) * 1e-6;

        evsim::EventPacket packet = m_eventGen->generate(
            m_lumaBuf.data(), static_cast<int>(w), static_cast<int>(h),
            static_cast<int>(w), timestampSec);

        publishEvents(packet, cameraIndex);
    }

    void onRenderHelper(dwCameraFrameHandle_t frame, uint8_t cameraIndex)
    {
        dwImageHandle_t img = DW_NULL_HANDLE;
        dwCameraOutputType outputType = DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8;

        CHECK_DW_ERROR(dwRenderEngine_setTile(m_tileVideo[cameraIndex], m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));

        CHECK_DW_ERROR(dwSensorCamera_getImage(&img, outputType, frame));

        // stream that image to the GL domain
        CHECK_DW_ERROR(dwImageStreamerGL_producerSend(img, m_streamerToGL[cameraIndex]));

        // receive the streamed image as a handle
        dwImageHandle_t frameGL;
        CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&frameGL, 33000, m_streamerToGL[cameraIndex]));

        dwImageGL *imageGL;
        CHECK_DW_ERROR(dwImage_getGL(&imageGL, frameGL));

        // render received texture
        {
            dwVector2f range{};
            range.x = imageGL->prop.width;
            range.y = imageGL->prop.height;
            CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
            CHECK_DW_ERROR(dwRenderEngine_renderImage2D(imageGL, {0, 0, range.x, range.y}, m_renderEngine));
            dwRenderEngine_setColor(m_colorPerPort[0], m_renderEngine);

            dwTime_t timestamp;
            dwImage_getTimestamp(&timestamp, img);
            std::string tileString = std::string("Camera:") + std::to_string(cameraIndex) + std::string(" Time:") + std::to_string(timestamp);

            CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine));
            CHECK_DW_ERROR(dwRenderEngine_renderText2D(tileString.c_str(), {25, 80}, m_renderEngine));

            m_screenshot->processScreenshotTrig();
        }
        // returned the consumed image
        CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&frameGL, m_streamerToGL[cameraIndex]));

        // notify the producer that the work is done
        CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 32000, m_streamerToGL[cameraIndex]));

        if (m_frameGrab && img)
        {
            frameGrab(img, cameraIndex);
        }
    }

    void checkToPauseStartResetSensor()
    {
        if (m_isCameraSelected)
        {
            if (m_cameraSelected > m_totalCameras)
            {
                std::cout << "Index above camera count, no action" << std::endl;
                return;
            }

            if (m_pauseSensor)
            {
                std::cout << "Pausing sensor " << m_cameraSelected << std::endl;
                dwStatus status = dwSensor_stop(m_camera[m_cameraSelected]);
                if (status != DW_SUCCESS)
                {
                    throw std::runtime_error("onRender: Cannot pause camera " + std::to_string(m_cameraSelected));
                }
                m_pauseSensor = false;
            }

            if (m_startSensor)
            {
                std::cout << "Starting sensor " << m_cameraSelected << std::endl;
                dwStatus status = dwSensor_start(m_camera[m_cameraSelected]);
                if (status != DW_SUCCESS)
                {
                    throw std::runtime_error("onRender: Cannot start camera " + std::to_string(m_cameraSelected));
                }
                m_startSensor = false;
            }

            if (m_resetSensor)
            {
                std::cout << "Reset sensor " << m_cameraSelected << std::endl;
                dwStatus status = dwSensor_reset(m_camera[m_cameraSelected]);
                if (status != DW_SUCCESS)
                {
                    throw std::runtime_error("onRender: Cannot reset camera " + m_cameraSelected);
                }
                m_resetSensor = false;
            }
        }
    }

    void onRender() override
    {

        checkToPauseStartResetSensor();
        dwCameraFrameHandle_t frame[MAX_CAMS];

        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {
            dwStatus status = DW_NOT_READY;

            uint32_t countFailure = 0;
            while ((status == DW_NOT_READY) || (status == DW_END_OF_STREAM))
            {
                status = dwSensorCamera_readFrame(&frame[i], 333333, m_camera[i]);
                countFailure++;
                if (countFailure == 1000000)
                {
                    std::cout << "Camera doesn't seem responsive, exit loop and stopping the sample" << std::endl;
                    break;
                }

                // Reset the sensor to support loopback
                if (status == DW_END_OF_STREAM)
                {
                    dwSensor_reset(m_camera[i]);
                    std::cout << "Video reached end of stream" << std::endl;
                }
            }

            if (status == DW_TIME_OUT)
            {
                throw std::runtime_error("onRender: Timeout waiting for camera " + std::to_string(i));
            }

            CHECK_DW_ERROR(status);
        }
        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {

            if (m_recordCamera && i == 0)
            {
                if (m_bUseNewCodecStack)
                {
                    CHECK_DW_ERROR(dwSensorSerializerNew_serializeCameraFrameAsync(frame[i], m_serializerNew[dwCameraISPType::DW_CAMERA_ISP0]));
                }
                else
                {
                    CHECK_DW_ERROR(dwSensorSerializer_serializeCameraFrameAsync(frame[i], m_serializer[dwCameraISPType::DW_CAMERA_ISP0]));
                }
            }

            // for yuv420-uint8-bl or yuv420-uint16-bl we will skip the render of the current frame.
            if (m_useProcessed && m_enableRender[i])
            {
                onRenderHelper(frame[i], i);
            }

            // Event generation runs on the same frame, before it is returned.
            printf("onRender : start of process events\n");
            processEvents(frame[i], i);
#if (VIBRANTE_PDK_DECIMAL >= 6000200) || defined(LINUX_AND_EMU)
            dwImageHandle_t img = DW_NULL_HANDLE;
            if (m_useProcessed1)
            {
                if (m_recordCamera && m_isProcessed1WriteEnabled)
                {
                    if (m_bUseNewCodecStack)
                    {
                        CHECK_DW_ERROR(dwSensorSerializerNew_serializeCameraFrameAsync(frame[i], m_serializerNew[dwCameraISPType::DW_CAMERA_ISP1]));
                    }
                    else
                    {
                        CHECK_DW_ERROR(dwSensorSerializer_serializeCameraFrameAsync(frame[i], m_serializer[dwCameraISPType::DW_CAMERA_ISP1]));
                    }
                }

                // Only native processed ISP1/2 outputs(not cuda) are available for now
                CHECK_DW_ERROR(dwSensorCamera_getImage(&img, DW_CAMERA_OUTPUT_NATIVE_PROCESSED1, frame[i]));
            }
            if (m_useProcessed2)
            {
                CHECK_DW_ERROR(dwSensorCamera_getImage(&img, DW_CAMERA_OUTPUT_NATIVE_PROCESSED2, frame[i]));
            }
#endif
        }

        m_frameGrab = false;

        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {
            CHECK_DW_ERROR(dwSensorCamera_returnFrame(&frame[i]));
        }

        if (!m_useProcessed)
        {
            // Display a message on screen that no image is expected.
            CHECK_DW_ERROR(dwRenderEngine_setTile(m_tileVideo[0], m_renderEngine));
            CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));
            CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D({640, 480}, m_renderEngine));
            dwRenderEngine_setColor({1, 1, 1, 0}, m_renderEngine);
            CHECK_DW_ERROR(dwRenderEngine_renderText2D("No preview available when only raw capture.", {25, 25}, m_renderEngine));
        }

        renderEventTile();
        renderutils::renderFPS(m_renderEngine, getCurrentFPS());

        m_frameCount++;
        if ((m_frameCount % 500 == 0) && (m_serializer[dwCameraISPType::DW_CAMERA_ISP0] != nullptr))
        {
            logStats(dwCameraISPType::DW_CAMERA_ISP0); // serializerIdx = 0
        }
    }
};

// #######################################################################################
CameraCustomSimpleApp::CameraCustomSimpleApp(const ProgramArguments &args)
    : DriveWorksSample(args)
{
}

void CameraCustomSimpleApp::initializeDriveWorks(dwContextHandle_t &context) const
{
    // initialize logger to print verbose message on console in color
    CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
    CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));

    // initialize SDK context, using data folder
    dwContextParameters sdkParams = {};

#ifdef VIBRANTE
    sdkParams.eglDisplay = getEGLDisplay();
#endif

    CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, &sdkParams));
}

void cameraEventHandling(dwCameraSIPLNotification *notificationData, dwSensorHandle_t sensor)
{
    // skip the normal event, do not handle
    if (notificationData->data.eNotifyType == DW_NOTIF_INFO_ICP_PROCESSING_DONE || notificationData->data.eNotifyType == DW_NOTIF_INFO_ISP_PROCESSING_DONE || notificationData->data.eNotifyType == DW_NOTIF_INFO_ACP_PROCESSING_DONE || notificationData->data.eNotifyType == DW_NOTIF_INFO_CDI_PROCESSING_DONE)
    {
        return;
    }

    // output the error event
    std::cout << std::endl; // extra blank line
    std::cout << "--- Camera Device Error Info ---" << std::endl;
    std::cout << "Notification Type: " << notificationData->data.eNotifyType << std::endl;
    std::cout << "Pipeline ID: " << notificationData->data.uIndex << std::endl;
    std::cout << "dev block link mask: " << notificationData->data.uLinkMask << std::endl;
    std::cout << "Tsc Timestamp: " << notificationData->data.frameCaptureTSC << std::endl;
    std::cout << "GPIO indices:" << std::endl;
    for (uint32_t i = 0U; i < notificationData->data.numGpioIdxs; ++i)
    {
        std::cout << "    gpioIdxs[" << i << "]=" << notificationData->data.gpioIdxs[i] << std::endl;
    }

    // show the detailed info for deserializer failure
    if (notificationData->data.eNotifyType == DW_NOTIF_ERROR_DESERIALIZER_FAILURE)
    {
        std::cout << "deserializer info ---" << std::endl;
        std::cout << "sizeWritten: " << notificationData->deserializerErrorInfo.sizeWritten << std::endl;
        std::cout << "ErrorBuffer: ";
        for (auto i = 0U; i < notificationData->deserializerErrorInfo.sizeWritten; ++i)
        {
            std::cout << static_cast<unsigned char>(notificationData->deserializerErrorInfo.errorBuffer[i]);
        }
        std::cout << std::endl;
    }

    // show the detailed info for serializer/sensor failure
    if (notificationData->data.eNotifyType == DW_NOTIF_ERROR_SERIALIZER_FAILURE || notificationData->data.eNotifyType == DW_NOTIF_ERROR_SENSOR_FAILURE || (notificationData->data.eNotifyType == DW_NOTIF_ERROR_DESERIALIZER_FAILURE && notificationData->isRemoteError))
    {
        std::cout << "serializer info ---" << std::endl;
        for (auto i = 0U; i < notificationData->numCameraModules; ++i)
        {
            std::cout << "Camera: " << i << std::endl;
            std::cout << "serializer sizeWritten: " << notificationData->serializerErrorInfoList[i].sizeWritten << std::endl;
            std::cout << "serializer ErrorBuffer: ";
            for (auto j = 0U; j < notificationData->serializerErrorInfoList[i].sizeWritten; ++j)
            {
                std::cout << static_cast<unsigned char>(notificationData->serializerErrorInfoList[i].errorBuffer[j]);
            }
            std::cout << std::endl;
        }

        std::cout << "sensor info ---" << std::endl;
        for (auto i = 0U; i < notificationData->numCameraModules; ++i)
        {
            std::cout << "Camera: " << i << std::endl;
            std::cout << "sensor sizeWritten: " << notificationData->sensorErrorInfoList[i].sizeWritten << std::endl;
            std::cout << "sensor ErrorBuffer: ";
            for (auto j = 0U; j < notificationData->sensorErrorInfoList[i].sizeWritten; ++j)
            {
                std::cout << static_cast<unsigned char>(notificationData->sensorErrorInfoList[i].errorBuffer[j]);
            }
            std::cout << std::endl;
        }
    }

    std::cout << "--- End of the Error Info ---" << std::endl;
    if (isResetLinkOnError == true)
    {
        // reset the camera link
        dwSensorCamera_disableLink(sensor);
        std::cout << "Disable Link OK!" << std::endl;
        dwSensorCamera_enableLink(sensor, false);
        std::cout << "Enable Link OK!" << std::endl;
    }
}

// #######################################################################################
int main(int argc, const char **argv)
{
    // define all arguments used by the application
    ProgramArguments args(argc, argv,
                          {
                              ProgramArguments::Option_t("rig", (dw_samples::SamplesDataPath::get() + "/samples/sensors/camera/camera/rig.json").c_str(), "This is the main source of input configuration for the cameras, can setup any possible configuration"),
                              ProgramArguments::Option_t("write-file", "", "If this string is not empty, then the serializer will record in this location\n"),
                              ProgramArguments::Option_t("reset-on-error", "0", "If this flag set to 1, then the camera link will be reset when errors occur. "
                                                                                "This parameter only takes effect when --enable-user-event-callbacks=1"),
                              ProgramArguments::Option_t("enable-user-event-callbacks", "0", "If this set to 1, then the user-defined error handling will be enabled\n"),
                              ProgramArguments::Option_t("enable-processed1-write", "0", "If this set to 1, and write-file is not empty, serializer will record ISP Pipeline1 in the location\n"),
                              ProgramArguments::Option_t("enable-events", "0", "If set to 1, run the CUDA event-camera simulator (evsim) on one camera\n"),
                              ProgramArguments::Option_t("event-camera", "0", "Index of the camera feeding the event generator\n"),
                              ProgramArguments::Option_t("event-input", "raw", "Event generator input: 'raw' (NATIVE_RAW plane, GPU path) or 'rgba' (processed image converted to luma)\n"),
                              ProgramArguments::Option_t("event-raw-shift", "0", "Right-shift applied to raw samples (e.g. 4 for 12-bit data left-aligned in 16 bits)\n"),
                              ProgramArguments::Option_t("event-sensor", "DVS346", "Sensor preset passed to dvs::sensor_k_preset()\n"),
                              ProgramArguments::Option_t("event-deterministic", "1", "1 = fast deterministic kernel, 0 = stochastic kernel\n"),
                              ProgramArguments::Option_t("event-threshold-on", "1.0", "Contrast threshold for ON events\n"),
                              ProgramArguments::Option_t("event-threshold-off", "1.0", "Contrast threshold for OFF events\n"),
                              ProgramArguments::Option_t("event-max", "200000", "Max events per frame (fixed D2H copy size)\n"),
                              ProgramArguments::Option_t("event-csv", "", "If not empty, write events (x,y,t,polarity) to this CSV file\n"),
                              ProgramArguments::Option_t("event-display", "1", "If 1 (and --enable-events=1), show the latest event packet in an extra tile\n"),
                          },
                          "Sample illustrating how to use camera interfaces");

    // Window/GL based application
    CameraCustomSimpleApp app(args);
    app.initializeWindow("Camera Sample", 1280, 800, args.enabled("offscreen"));
    return app.run();
}
