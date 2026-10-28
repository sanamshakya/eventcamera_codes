## Update on EventGeneratorCUDA for profiling
- `EventGeneratorCUDA.cu`, `EventGeneratorCUDA.cuh` : Updated cuda kernel with deterministic and noise model with updated logic for profiling.
- `GenProfiler.h` :  Header handling profiling and printing the profile log

## EventGeneratorCUDA.cu updates
Added time capture at various stages of `EventPacket EventGeneratorCUDA::generate()` for profiling timing at each steps:

- host stage memcpy : The CPU copies the frame row by row from the adapter’s buffer into a pinned (page-locked) buffer, h_image_.
```
        const double tS0 = nowMs(); // [prof]
        for (int y = 0; y < height_; ++y)
        {
            std::memcpy(h_image_ + static_cast<size_t>(y) * width_,
                        image + static_cast<size_t>(y) * stridePixels,
                        static_cast<size_t>(width_) * sizeof(uint16_t));
        }
        prof_.hostStage = nowMs() - tS0; // [prof]

```
- Host to Device copy (H2D) : The DMA copy of that frame from pinned memory into GPU memory (d_image_). Measured using cudaEvents `ev[0]` to `ev[1]`
```
        cudaEventRecord(prof_.ev[0], stream_); // [prof] before H2D
        CUDA_CHECK(cudaMemcpyAsync(d_image_, h_image_, h2dBytes,
                                   cudaMemcpyHostToDevice, stream_));
        cudaEventRecord(prof_.ev[1], stream_); // [prof] after H2D
```
- CUDA kernel run : From the end of the H2D copy to the end of the event kernel. Measured using cudaEvents stored in `ev[1]` and `ev[2]`
```
  else
        {
            GenerateEventsKernelDVSStochastic<<<grid, block, 0, stream_>>>(
                d_pixelStates_, d_rngStates_, d_image_, width_, height_, width_,
                static_cast<float>(k_.k1), static_cast<float>(k_.k2), static_cast<float>(k_.k3),
                static_cast<float>(k_.k4), static_cast<float>(k_.k5), static_cast<float>(k_.k6),
                static_cast<float>(config_.contrastThresholdOn),
                static_cast<float>(config_.contrastThresholdOff),
                static_cast<float>(dtFrameUs), dtFrameSeconds, previousTimestamp_,
                maxEvents_, d_outputBuffer_);
        }
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(prof_.ev[2], stream_); // [prof] after kernel
        //ev[1] already stored above
```
- Device to Host(D2H) copy : From the end of the kernel to the end of the second copy.

```
        // Step 2: only the events actually produced
        const size_t d2hBytes = sizeof(int) + static_cast<size_t>(eventCount) * sizeof(Event);
        if (eventCount > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(h_outputBuffer_ + kHeaderBytes,
                                       d_outputBuffer_ + kHeaderBytes,
                                       static_cast<size_t>(eventCount) * sizeof(Event),
                                       cudaMemcpyDeviceToHost, stream_));
        }
        cudaEventRecord(prof_.ev[3], stream_); // after D2H
```

- Host blocked in sync : The time the CPU spent waiting inside cudaStreamSynchronize (the first one, after enqueuing the count copy). This is the time CPU waiting for CUDA async tasks are completed. It is like `wait()` in threaded programming.
```
const double tY0 = nowMs();
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        int eventCount = *reinterpret_cast<const int *>(h_outputBuffer_);
        if (eventCount > maxEvents_)
        {
            fprintf(stderr, "EventGeneratorCUDA: %d events, capacity %d - %d dropped.\n",
                    eventCount, maxEvents_, eventCount - maxEvents_);
            eventCount = maxEvents_;
        }

        // Step 2: only the events actually produced
        const size_t d2hBytes = sizeof(int) + static_cast<size_t>(eventCount) * sizeof(Event);
        if (eventCount > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(h_outputBuffer_ + kHeaderBytes,
                                       d_outputBuffer_ + kHeaderBytes,
                                       static_cast<size_t>(eventCount) * sizeof(Event),
                                       cudaMemcpyDeviceToHost, stream_));
        }
        cudaEventRecord(prof_.ev[3], stream_); // after D2H
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        prof_.hostSync = nowMs() - tY0;
```

- packet.assign : The CPU copies the events out of the pinned buffer into a std::vector<Event> inside the returned EventPacket.
```
        const double tA0 = nowMs();                              // [prof]
        packet.events.assign(devEvents, devEvents + eventCount); // O(eventCount), not O(W*H)
        prof_.hostAssign = nowMs() - tA0;                        // [prof]

```
- total time in event generation per frame : Calculated by subtracting timestamp at beginning and end of generate() function
```
    EventPacket EventGeneratorCUDA::generate(
        const uint16_t *image,
        int width,
        int height,
        int stridePixels,
        double timestamp)
    {
        EventPacket packet{};
        if (image == nullptr)
            return packet;
        const double tFn0 = nowMs(); // [prof] start of generate()
        ....
        ....
        prof_.hostTotal = nowMs() - tFn0; // [prof]
        prof_.endFrame(static_cast<size_t>(eventCount), d2hBytes, h2dBytes);
        return packet;
    }

```
## Update in `EventGeneratorCUDA.cuh`
- Added header for `GenProfiler.h`
```
#include "Config.h"
#include "Event.h"
#include "EventPacket.h"
#include "IEventGenerator.h"
#include "GenProfiler.h" //new header
```
- Initialize the profiler :
```
...
...
    size_t outputBufferBytes_ = 0;

    GenProfiler prof_{100, 5}; // report every 100 frames, ignore first 5
};
```

## Build steps : 
- Add the updated cuda kernel code and `GenProfiler.h`
- Next build the cuda kernel with nvcc build command

## Console log after running above updated kernel : 
```
...
...
Received packet with 3311 events
Buffer size = 7467
Drawing 5647 events
[gen-prof] 100 frames, 2132 events/frame avg
  host stage memcpy      avg    0.382  max    0.623 ms
  H2D (gpu)              avg    0.289  max    0.291 ms
  kernel (gpu)           avg    3.224  max    3.695 ms
  D2H (gpu)              avg    0.400  max    5.690 ms
  host blocked in sync   avg    4.043  max    9.189 ms
  packet.assign          avg    0.158  max    0.557 ms
  generate() TOTAL       avg    4.955  max   10.052 ms
  D2H 0.06 MB @ 0.14 GB/s | H2D 0.61 MB @ 2.13 GB/s
  share of total: H2D 6%  kernel 65%  D2H 8%  stage 8%  assign 3%
Received packet with 2351 events
Buffer size = 7998
...
...

```