// CIntensityCaptureWriter.hpp
//
// Dumps the per-frame intensity buffer - the same downscaled
// uint16_t[eventFrameWidth * eventFrameHeight] grid that
// EventGenerator::generate()/EventGeneratorCUDA::generate() actually
// consumes each frame - to one CSV file per frame, as a plain
// width x height grid (one row per image row, comma-separated columns,
// no header). Lets you line a frame's raw intensities up against the
// events.csv/frames.csv rows for the same frame number (see
// EventLogger.h) when debugging threshold behavior.
//


#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

class CIntensityCaptureWriter
{
public:
    // pathPrefix: files are written as "<pathPrefix>_frame_<N>.csv",
    // where N is the frame index passed to WriteFrame() 
    void Init(const std::string &pathPrefix, uint32_t numFrames)
    {
        pathPrefix_ = pathPrefix;
        maxFrames_ = numFrames;
        written_ = 0;
        enabled_ = true;
    }

    bool IsEnabled() const { return enabled_; }

    // True once numFrames files have been written - caller can use this
    // to skip the (otherwise wasted) call entirely once capture is done.
    bool IsDone() const { return !enabled_ || (written_ >= maxFrames_); }

    // data: width*height values, row-major, row stride = stridePixels
    
    bool WriteFrame(uint64_t frameIndex, const uint16_t *data,
                    int width, int height, int stridePixels)
    {
        if (!enabled_ || written_ >= maxFrames_ || data == nullptr)
            return false;

        const std::string path =
            pathPrefix_ + "_frame_" + std::to_string(frameIndex) + ".csv";

        FILE *f = std::fopen(path.c_str(), "w");
        if (f == nullptr)
            return false;

        // Build one row at a time into a reusable text buffer, then a
        // single fwrite per row - keeps this well clear of fprintf's
        // per-value formatting/locale overhead at ~1M values/frame while
        // staying simple 
        rowBuf_.clear();
        rowBuf_.reserve(static_cast<size_t>(width) * 6); // "65535," worst case per value

        for (int y = 0; y < height; ++y)
        {
            const uint16_t *row = data + static_cast<size_t>(y) * stridePixels;

            rowBuf_.clear();
            for (int x = 0; x < width; ++x)
            {
                char tmp[8];
                const int n = std::snprintf(tmp, sizeof(tmp), (x + 1 < width) ? "%u," : "%u",
                                             static_cast<unsigned>(row[x]));
                rowBuf_.insert(rowBuf_.end(), tmp, tmp + n);
            }
            rowBuf_.push_back('\n');

            std::fwrite(rowBuf_.data(), 1, rowBuf_.size(), f);
        }

        std::fclose(f);
        ++written_;
        return true;
    }

private:
    std::string pathPrefix_;
    uint32_t maxFrames_ = 0;
    uint32_t written_ = 0;
    bool enabled_ = false;

    std::vector<char> rowBuf_; // reused across rows/frames - no per-row allocation
};
