#  EventLogger class for saving events csv file
Steps for integrating event logger class into nvispl camera app with CUDA based event generation
- Add the `EventLogger.cpp` and `EventLogger.h` file into previously build nvsipl camera CUDA based event generation project
- Update `CNvSIPLConsumer.hpp` to integrate `EventLogger` 
    - Add the event logging enabling function as previously done for event generation
    ```
    void EnableEventCSVLogging(const string &sPathPrefix, const string &sLabel = "capture")
    {
        m_pEventLogger.reset(new evsim::EventLogger(sPathPrefix, sLabel));
    }
    ```

    - Call the event logger logging function at `EventProcessingThreadFunc` for logging the event packets
    ```
    // --- CSV logging (events.csv + frames.csv) ---
            // One frames.csv row per frame (even zero-event ones) and one
            // events.csv row per event, written from this one worker
            // thread only - see EnableEventCSVLogging().
            if (m_pEventLogger != nullptr)
            {
                for (const auto &e : packet.events)
                {
                    m_pEventLogger->log(static_cast<int>(packet.frameNumber), e.timestamp,
                                         static_cast<int>(e.x), static_cast<int>(e.y),
                                         static_cast<int>(e.polarity));
                }
                m_pEventLogger->logFrameCount(static_cast<int>(packet.frameNumber),
                                              packet.endTime,
                                              static_cast<int>(packet.events.size()));
            }
    ```
    - Add the event logger cleanup functions at `Deinit()`
    ```
     if (m_pEventLogger != nullptr)
        {
            m_pEventLogger->flush();
            m_pEventLogger = nullptr;
        }

    ```
- Update in Makefile
    - Add the object file generation for EventLogger.cpp by adding following line : 
    ```
    OBJS   += main.o
    OBJS   += EventGenerator.o
    OBJS   += EventLogger.o
    ```

- Update in main.cpp
    - Enable the event logging in the main.cpp as done for event generator
    ```
    upCons->EnableEventGeneration(evConfig,"events.bin");
    upCons->EnableEventCSVLogging("capture", "capture");
    ```

- After above updates, build and run the application
- After application is stopped, two csv files `capture.events.csv` and `captre.frames.csv` will be created

- Next comparing two event captures.
    - Run the application two times and rename the captured csv files and csv1 and csv2
    - Next run the `scripts/compare_events_plot.py` to plot and compare two events metrics
    ```
    python3 compare_events_plot.py capture1 capture2
    ```
    For above test `capture1.events.csv, capture1.frames.csv and capture2.events.csv, capture2.frames.csv` must be present in same script directory. 
    After running the python script two plots for event rate and per pixel inter event interval file will be generated.
    
## Update for saving raw input data
- `CIntensityCaptureWriter` class is integrated into `CNvSIPLConsumer.hpp` to capture first 50 raw input frames.
-  Add `CIntensityCaptureWriter.hpp` file into nvsipl camera with CUDA based event generation application
- In main.cpp, enable the raw input data caputre by adding the followings after the EnableEventsCSVLogging: 
```
 upCons->EnableEventGeneration(evConfig,"events.bin");
 upCons->EnableEventCSVLogging("capture", "capture");
 upCons->EnableIntensityCSVCapture("intensity", 50);
 
```
- After building and running the application with changes, raw intensity frames are captured as `intensity_frame_X.csv`.
- Next analyze and plot histogram of the raw captured intensity file by running :
```
//for analysing single frame
python3 analyze_intensity_csv.py --prefix intensity --frames 1

//for analysing all frames
python3 analyze_intensity_csv.py --prefix intensity
```


    
