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