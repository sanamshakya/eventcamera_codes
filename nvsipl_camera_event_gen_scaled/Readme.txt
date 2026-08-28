## Major Changes

- Instead of taking partial data by using ROI from RAW Bayer BGGR frame, downscale the whole raw data removing the metadata and pass to event generation queue 
```
...
 std::vector<uint16_t> raw16;
 int rawWidth = 0, rawHeight = 0, rawStride = 0;
 int activeTopRow = 0, activeHeight = 0;
...
 const uint16_t *activeTop = raw16.data() + static_cast<size_t>(activeTopRow) * rawStride;

 std::vector<uint16_t> eventFrame;
        DownscaleRaw16Box3x3(activeTop, rawStride,
                              eventFrameWidth, eventFrameHeight, eventFrame);

        // eventFrame is tightly packed at 1280 wide -> stride == width
        EnqueueRawFrame(std::move(eventFrame), eventFrameWidth,
                         eventFrameHeight, /*stridePixels=*/eventFrameWidth,
                         timestamp);
                         
...
                          
 
```
