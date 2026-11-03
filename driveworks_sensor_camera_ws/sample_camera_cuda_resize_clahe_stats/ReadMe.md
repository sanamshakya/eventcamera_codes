## Driveworks sample camera application with CUDA event generator

- Copy and replace the camera sample application located at `<driveworks src>/samples/src/sensors/camera/camera` with the current source content in this repo
- Build the appication by running cmake at the sample main directory :
```
cd <driverworks src>/samples
mkdir build
cd build
cmake ..
make VERBOSE=1 sample_camera
```
Check that EventGeneratorCUDA.cu is compiled by nvcc with -arch=sm_87 (or --generate-code=arch=compute_87,code=sm_87) in the build log.

## Updates for Event logger integration
- Added `AsyncEventLogger` in the main.cpp for handling new event logger class
- Added `std::unique_ptr<AsyncEventLogger> m_eventLogger;` handler for event logging class
- Added command line argument for event logger file 
```
// --event-csv is a path PREFIX: EventLogger appends ".events.csv" and
                // ".frames.csv". A trailing ".csv" is stripped so "out/ev.csv" works too.
                std::string csvPath = getArgument("event-csv");
```
-    In publish events 
```
     // Last: hands the events to the writer thread (moves them out of `packet`).
        if (m_eventLogger)
            m_eventLogger->push(std::move(packet));
```

## Updates in EventGeneratorCUDA kernel for input stats:
- Added input stats code for input timestamps and intensity values, enabled by EVSIM_INPUT_STATS=1 environment variable
```
// Debug only (env EVSIM_INPUT_STATS=1): intensity range the kernels really see
        // (after the shift) and the frame interval. The sensor K presets are
        // calibrated for roughly 0..255 intensities; a raw range of 0..4095 or
        // 0..65535 inflates the leak term k5*avg*dt and with it the event count.
        static const bool inputStats = std::getenv("EVSIM_INPUT_STATS") != nullptr;
        if (inputStats && (static_cast<unsigned long long>(frameNumber_) % 30ull) == 0ull)
        {
            const size_t n = static_cast<size_t>(width_) * height_;
            CUDA_CHECK(cudaMemcpyAsync(h_image_, d_image_, n * sizeof(uint16_t),
                                       cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            uint16_t mn = 0xFFFF, mx = 0;
            double sum = 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                const uint16_t v = h_image_[i];
                mn = std::min(mn, v);
                mx = std::max(mx, v);
                sum += v;
            }
            fprintf(stderr, "[input] frame %llu: shift %d -> min %u max %u mean %.1f | dt %.3f ms\n",
                    static_cast<unsigned long long>(frameNumber_), rawShift, mn, mx, sum / n,
                    initialized_ ? (timestamp - previousTimestamp_) * 1e3 : 0.0);
        }
```

## Driveworks Sample Camera code flow
- In main.cpp
    - Handle based application for configuring and running the camera pipeline. Handles declared at Class declaration :
    ```
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
            //   raw  (default): DW_CAMERA_OUTPUT_CUDA_RAW_UINT16 is already a CUDA image.
            //                   Its device pointer goes straight to generateFromDevice()
            //                   (GPU->GPU repack + optional bit shift) -- no streamer
            //                   (cuda->cuda streamers are cross-process only), no host
            //                   copy. Needs the CUDA raw output in the rig's
            //                   output-requested.
            //   rgba          : CUDA RGBA8 (processed) image -> host luma -> generate().
            std::unique_ptr<evsim::EventGeneratorCUDA> m_eventGen;
            bool m_eventUseRaw = true;
            int m_eventRawShift = 0;
            uint32_t m_eventCamera = 0;
            std::vector<uint16_t> m_lumaBuf;
            std::unique_ptr<AsyncEventLogger> m_eventLogger; // --event-csv: <prefix>.events.csv / .frames.csv
            // Event time base. DriveWorks timestamps are int64 microseconds (absolute,
            // ~1.7e15 us). Event/CSV time is seconds RELATIVE to the first event-camera
            // frame (like the nvsipl "capture" log), computed on the integer us value so
            // nothing is lost to double rounding. --event-time-origin=absolute keeps the
            // raw DriveWorks timestamp instead.
            dwTime_t m_evT0Us = -1; // timestamp of the first event-camera frame
            bool m_evAbsoluteTime = false;
            double m_evPrevTsSec = -1.0;                     // previous frame timestamp (diagnostics)
            uint64_t m_evBadDt = 0;                          // frames with non-increasing timestamp
            uint64_t m_eventFrames = 0;
            uint64_t m_eventTotal = 0;
            size_t m_eventLastPacket = 0;

            // ---- event display: accumulated ON/OFF count image ----
            // Events are accumulated into per-pixel counters and shown as one RGBA
            // texture (CPU image -> GL streamer -> renderImage2D). The upload size is
            // fixed (W*H*4 bytes) regardless of event rate, so the render engine's
            // vertex buffer is never exceeded (no DW_OUT_OF_BOUNDS).
            bool m_eventDisplay = false; // extra tile showing the accumulated events
            uint32_t m_eventW = 0;
            uint32_t m_eventH = 0;
            std::vector<uint16_t> m_evOnCnt;                           // W*H ON counters
            std::vector<uint16_t> m_evOffCnt;                          // W*H OFF counters
            dwImageHandle_t m_evImg = DW_NULL_HANDLE;                  // CPU RGBA8 image
            dwImageStreamerHandle_t m_streamerEvToGL = DW_NULL_HANDLE; // CPU -> GL
            int m_evFramesPerWin = 1;                                  // accumulate this many frames per refresh
            int m_evFrameCount = 0;
            float m_evGain = 64.0f; // counts -> 0..255 brightness (1 count = 64)
            size_t m_evOnLast = 0;  // ON events in the last packet (HUD)
            size_t m_evOffLast = 0; // OFF events in the last packet (HUD)

            // uint32_t m_tileVideo[MAX_CAMS + 1];
            uint32_t m_srcW = 0, m_srcH = 0; // native camera size
            bool m_needResize = false;
            uint16_t *m_rawResized = nullptr;            // device buffer, m_eventW*m_eventH uint16 (raw path)
            std::vector<int> m_xi0, m_xi1, m_yi0, m_yi1; // host resize tables (rgba path)
            std::vector<float> m_xw, m_yw;
    ```
    - Main `class CameraCustomSimpleApp` life cycle is handled by following :
        - onInitialize()
            - for initializing handles for `m_context`, camera sensor, rig, visualization, initialize streamers and CUDA based Event Genrators
        - onProcess() 
            - Currently empty
        - onRelease()
            - for cleaning up the memory and releasing handles
        - onRender()
            - Capture frame from camera and pass to visualization and Event Generation
            - Image capture, pre-processing (in case of RGBA input) and calling Event Generation CUDA kernel done by calling 
            ```
                // Event generation runs on the same frame, before it is returned.
                processEvents(frame[i], i);
            ```
        - void processEvents(dwCameraFrameHandle_t frame, uint32_t cameraIndex) :
            - Based on the input type 'RAW' or 'RGBA' two pipeline
                - For 'RAW' input, call `processEventsRaw()`
                ```
                    if (m_eventUseRaw)
                    {
                        processEventsRaw(frame, cameraIndex);
                        return;
                    }
                ```
                - For 'RGBA' input,
                    - get the CUDA RGBA8 image of the frame
                    ```
                        dwImageHandle_t img = DW_NULL_HANDLE;
                        CHECK_DW_ERROR(dwSensorCamera_getImage(&img, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame));

                        dwTime_t timestampUs = 0;
                        CHECK_DW_ERROR(dwImage_getTimestamp(&timestampUs, img));

                        
                    ```
                    - image is capture in `imgCPU `
                    - stream it to the CPU 
                    ```
                        const dwTime_t timeout = 500000;
                        CHECK_DW_ERROR(dwImageStreamer_producerSend(img, m_streamerToCPUGrab[cameraIndex]));

                        dwImageHandle_t frameCPU = DW_NULL_HANDLE;
                        CHECK_DW_ERROR(dwImageStreamer_consumerReceive(&frameCPU, timeout, m_streamerToCPUGrab[cameraIndex]));

                        dwImageCPU *imgCPU = nullptr;
                        CHECK_DW_ERROR(dwImage_getCPU(&imgCPU, frameCPU));
                    ```
                    - convert to a luma plane (uint16, 0..255)
                    ```
                        const uint8_t *base = imgCPU->data[0];
                        const size_t pitch = imgCPU->pitch[0];

                        if (!m_needResize)
                        {
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
                        }
                        else
                        {
                            for (uint32_t y = 0; y < m_eventH; ++y)
                            {
                                const uint8_t *r0 = base + size_t(m_yi0[y]) * pitch;
                                const uint8_t *r1 = base + size_t(m_yi1[y]) * pitch;
                                const float wy = m_yw[y];
                                uint16_t *dst = m_lumaBuf.data() + size_t(y) * m_eventW;
                                for (uint32_t x = 0; x < m_eventW; ++x)
                                {
                                    const size_t a = size_t(m_xi0[x]) * 4, b = size_t(m_xi1[x]) * 4;
                                    const float wx = m_xw[x];
                                    const float top = (1.f - wx) * luma(r0 + a) + wx * luma(r0 + b);
                                    const float bot = (1.f - wx) * luma(r1 + a) + wx * luma(r1 + b);
                                    dst[x] = static_cast<uint16_t>((1.f - wy) * top + wy * bot + 0.5f);
                                }
                            }
                        }
                    ```
                    - run EventGeneratorCUDA::generate()
                    ```
                        // m_lumaBuf is tightly packed at the EVENT size (also when resizing), so pass
                        // the event size and stride -- not the native camera size.
                        evsim::EventPacket packet = m_eventGen->generate(
                            m_lumaBuf.data(), static_cast<int>(m_eventW), static_cast<int>(m_eventH),
                            static_cast<int>(m_eventW), timestampSec);
                    ```
        - void processEventsRaw(dwCameraFrameHandle_t frame, uint32_t cameraIndex)
            - Capture RAW frames and pass image frame pointer from CUDA->CUDA no streamer required
            ```
                dwImageHandle_t raw = DW_NULL_HANDLE;
                CHECK_DW_ERROR(dwSensorCamera_getImage(&raw, DW_CAMERA_OUTPUT_CUDA_RAW_UINT16, frame));

                dwTime_t timestampUs = 0;
                CHECK_DW_ERROR(dwImage_getTimestamp(&timestampUs, raw));

                dwImageCUDA *imgCUDA = nullptr;
                CHECK_DW_ERROR(dwImage_getCUDA(&imgCUDA, raw));

            ``` 
            image frame is stored in `imgCUDA`
            -  Captured frame is passed to the CUDA event generation kernel
                - if resizing enabled `launchResizeRawU16` is called before calling `m_eventGen->generateFromDevice`
                ```
                    evsim::EventPacket packet;

                    const uint16_t *src = reinterpret_cast<const uint16_t *>(imgCUDA->dptr[0]);
                    const int w = static_cast<int>(imgCUDA->prop.width);
                    const int h = static_cast<int>(imgCUDA->prop.height);
                    const size_t pitch = static_cast<size_t>(imgCUDA->pitch[0]);
                    const double tSec = timestampSec; // same time base as checkEventTimestamp()/the logger

                    if (m_needResize)
                    {
                        if (w != static_cast<int>(m_srcW) || h != static_cast<int>(m_srcH))
                            throw std::runtime_error("processEventsRaw: frame size changed after initialization");

                        const cudaError_t e = launchResizeRawU16(src, pitch, w, h, m_rawResized,
                                                                static_cast<int>(m_eventW), static_cast<int>(m_eventH));
                        if (e != cudaSuccess)
                            throw std::runtime_error(std::string("raw resize failed: ") + cudaGetErrorString(e));

                        packet = m_eventGen->generateFromDevice(m_rawResized,
                                                                static_cast<int>(m_eventW), static_cast<int>(m_eventH),
                                                                size_t(m_eventW) * sizeof(uint16_t), // tightly packed
                                                                tSec, m_eventRawShift);
                    }
                    else
                    {
                        packet = m_eventGen->generateFromDevice(src, w, h, pitch, tSec, m_eventRawShift);
                    }
                ```

            - 

## CMakeLists.txt updates
- adding source files for CUDA kernel and event logging *.cpp at SOURCE variable 
```
...
set(SOURCES
    main.cpp
    EventLogger.cpp
    ClaheCuda.cu 
    ResizeRaw.cu 
    EventGeneratorCUDA.cu
)
...
```

## Build steps
- Copy and replace the code at `/usr/local/driverworks/samples/src/sensors/camera/camera` contained in this repo
- After code update, go to `/usr/local/driverworks/samples`
- create `build` directory and run cmake command with parameter for building for Orin architecture :
```
cd build
cmake ../ -DCMAKE_TOOLCHAIN_FILE=/usr/local/driveworks/samples/cmake/Toolchain-V5L.cmake

```
- Run make command for building driveworks `sample_camera` application
```
make sample_camera
```
- This will create `sample_camera` application in build directory at `<build dir>/src/sensors/camera/camera/`
- Copy the built application to the Dev kit. Here is the example using `scp`
```
scp src/sensors/camera/camera/sample_camera nvidia@192.168.1.1:
```

## Test and run
- Command line parameters for driveworks `sample_camera` application:
```
--rig: rig.json file for camera hardware pipeline configuration
--enable-events : Set to 1 for running event generation 
--event-display : Set to 1 to display event frames
--event-input : Set to 'raw' or 'rgba' depending upon available input stream from camera
--event-deterministic : Set to '1' for fast deterministic CUDA kernel without noise modeling and set to '0' for stocastic CUDA kernel
--event-max : Controls memory for maximum event count per frame
--event-sensor : K parameter presets as sensor types stored in 'dvs_types.hpp`
--event-clahe : Set to 1 to enable CLAHE CUDA kernel based processing for event generator input. Toggle at runtime with 'H' key
--event-csv : if set with <prefix>, log events with <prefix>.events.csv and <prefix>.frames.csv
```
More other options can be found by running `./sample camera --help`

-Commands for running in `RAW` and `RGBA` mode along with capturing event packets in *.csv files
```
    #for raw clahe
    EVSIM_INPUT_STATS=1 ./sample_camera --rig rig_raw.json --enable-events=1 --event-display=1 --event-input=raw --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_raw --event-clahe=1 --event-height=720 --event-width=1280 --event-csv=clahe_raw_custom_sensor >test_drivework_visualize.log

    #for raw rgba
    EVSIM_INPUT_STATS=1 ./sample_camera --rig rig.json --enable-events=1 --event-display=1 --event-input=rgba --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_rgba --event-clahe=1 --event-height=720 --event-width=1280 --event-csv=clahe_rgba_custom_sensor >test_drivework_visualize.log

    #for default raw (without clahe)
    EVSIM_INPUT_STATS=1 ./sample_camera --rig rig_raw.json --enable-events=1 --event-display=1 --event-input=raw --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_raw --event-clahe=0 --event-height=720 --event-width=1280 --event-csv=default_raw_custom_sensor >test_drivework_visualize.log

    #for default rgba (without clahe)
    EVSIM_INPUT_STATS=1 ./sample_camera --rig rig.json --enable-events=1 --event-display=1 --event-input=rgba --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_rgba --event-clahe=0 --event-height=720 --event-width=1280 --event-csv=default_rgba_custom_sensor >test_drivework_visualize.log

```

- Above run will display RGBA and Event frame side by side along with input statistics and CUDA kernel profiling in the terminal and creates `*.events.csv and *.frames.csv` are created with suffix as set for `--event-csv` in command line argument



## DVS sensor model parameters 
In current sensor model, event output depends upon three parameters :
```
drift  = K1 * rate of change in intensity + K4 + K5 * (average intensity)
Here K4 : constant drift (dark current) term,
and K5 : illumination-dependent drift term

modeled actual change in intensity = previous_residual_value + drift * delta_time

```
So even if rate of change in intensity = 0, there will be drift in intesity due to    K4 + K5 * (average intensity) term. 
So setting K4 and K5 to zero will remove the event accumulation due to these parameter and decrease noise when there is no any intesity change or motion in camera frame.

These parameters are stored in `dvs_types.h` file. Corresponding to `sensorType` as `Raw2DVS346` in `Config.h` corresponding camera model parameters in `dvs_types.h`.
```
if (camera_type == "Raw2DVS346")
        return SensorK{2.388, 4.166e-7, 1.541e-6, 9.768e-8, 1.466e-11, 9.824e-6};
//So in current config setting k4 and k5 terms to zero for removing DVS sensor's dark current / illumination-dependent leakage above line becomes :
		return SensorK{2.388, 4.166e-7, 1.541e-6, 0.0, 0.0, 9.824e-6};		
```



## Plotting and profiling captured event csv files
- Scripts for profiling are located at `<repo>/event_data_compare_test`
- For multi event captures comparison run, just giving the first filename of event csv files (for example ) :
```
    python3 ./compare_events_plot_multi_ev_rate.py <event file1 first name> <event file2 first name> ...
    python3 ./compare_events_plot_multi_ev_rate.py 
    clahe_rgba_custom_sensor clahe_raw_custom_sensor_k1_7 clahe_rgba_custom_sensor_deterministic clahe_raw_custom_sensor_deterministic
```
- This will generate two plots, `compare_tau.png` and `compare_rate.png`
    - `compare_tau.png` - plots inter event time interval, time since last event at same pixel as probability density function
    - `compare_rate.png` - plots events count per 100 msec time binning, which helps to visualize event rates


