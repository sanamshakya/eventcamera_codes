#include "EventGeneratorCUDA.cuh"
#include <cstdio>
#include <cstring>
#include <stdexcept>

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t err_ = (expr);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, \
                    __LINE__, cudaGetErrorString(err_));                     \
            throw std::runtime_error("CUDA call failed");                    \
        }                                                                    \
    } while (0)

namespace evsim
{
namespace
{

__global__ void InitFirstFrameKernel(PixelStateGPU *states,
                                      const uint16_t *image,
                                      int width, int height, int stride,
                                      double timestamp)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float intensity = static_cast<float>(image[static_cast<size_t>(y) * stride + x]);
    states[y * width + x] = PixelStateGPU{ intensity, timestamp };
}

// One thread per pixel, no cross-pixel dependencies -> fully parallel.
// Each pixel reserves its own output slots with a single atomicAdd (not
// one per event), so contention is one atomic op per *firing* pixel, not
// per event -- cheap even under heavy motion.
__global__ void GenerateEventsKernel(PixelStateGPU *states,
                                      const uint16_t *image,
                                      int width, int height, int stride,
                                      double timestamp,
                                      float positiveThreshold,
                                      float negativeThreshold,
                                      int maxEvents,
                                      uint8_t *outputBuffer /* [int count][Event...] */)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    PixelStateGPU state = states[idx];

    const float currentIntensity = static_cast<float>(image[static_cast<size_t>(y) * stride + x]);
    const float delta = currentIntensity - state.referenceIntensity;

    int *eventCount = reinterpret_cast<int *>(outputBuffer);
    Event *events = reinterpret_cast<Event *>(outputBuffer + sizeof(int));

    if (delta >= positiveThreshold)
    {
        const int crossings = static_cast<int>(floorf(delta / positiveThreshold));
        if (crossings > 0)
        {
            const int base = atomicAdd(eventCount, crossings);
            const float invDelta = 1.0f / delta;
            for (int i = 0; i < crossings; ++i)
            {
                const int slot = base + i;
                if (slot >= maxEvents) break; // dropped -- host logs overflow

                const float crossingValue = state.referenceIntensity + (i + 1) * positiveThreshold;
                float alpha = (crossingValue - state.referenceIntensity) * invDelta;
                alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
                const double eventTime = state.referenceTime + alpha * (timestamp - state.referenceTime);

                events[slot] = Event{ static_cast<uint16_t>(x), static_cast<uint16_t>(y),
                                       eventTime, +1 };
            }
            state.referenceIntensity += crossings * positiveThreshold;
            state.referenceTime = timestamp;
        }
    }
    else if (delta <= negativeThreshold)
    {
        const int crossings = static_cast<int>(floorf((-delta) / (-negativeThreshold)));
        if (crossings > 0)
        {
            const int base = atomicAdd(eventCount, crossings);
            const float invNegDelta = 1.0f / (-delta);
            for (int i = 0; i < crossings; ++i)
            {
                const int slot = base + i;
                if (slot >= maxEvents) break;

                const float crossingValue = state.referenceIntensity - (i + 1) * (-negativeThreshold);
                float alpha = (state.referenceIntensity - crossingValue) * invNegDelta;
                alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
                const double eventTime = state.referenceTime + alpha * (timestamp - state.referenceTime);

                events[slot] = Event{ static_cast<uint16_t>(x), static_cast<uint16_t>(y),
                                       eventTime, -1 };
            }
            state.referenceIntensity -= crossings * (-negativeThreshold);
            state.referenceTime = timestamp;
        }
    }

    states[idx] = state;
}

} // anonymous namespace

EventGeneratorCUDA::EventGeneratorCUDA(const Config &config, int maxEventsPerFrame)
    : config_(config), maxEvents_(maxEventsPerFrame)
{
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

EventGeneratorCUDA::~EventGeneratorCUDA()
{
    freeDeviceBuffers();
    if (stream_) cudaStreamDestroy(stream_);
}

void EventGeneratorCUDA::freeDeviceBuffers()
{
    if (d_image_)        cudaFree(d_image_);
    if (d_pixelStates_)  cudaFree(d_pixelStates_);
    if (d_outputBuffer_) cudaFree(d_outputBuffer_);
    if (h_image_)        cudaFreeHost(h_image_);
    if (h_outputBuffer_) cudaFreeHost(h_outputBuffer_);
    if (h_pixel0_)       cudaFreeHost(h_pixel0_);
    d_image_ = nullptr; d_pixelStates_ = nullptr; d_outputBuffer_ = nullptr;
    h_image_ = nullptr; h_outputBuffer_ = nullptr; h_pixel0_ = nullptr;
}

void EventGeneratorCUDA::allocateDeviceBuffers()
{
    const size_t numPixels = static_cast<size_t>(width_) * height_;
    outputBufferBytes_ = sizeof(int) + static_cast<size_t>(maxEvents_) * sizeof(Event);

    CUDA_CHECK(cudaMalloc(&d_image_, numPixels * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_pixelStates_, numPixels * sizeof(PixelStateGPU)));
    CUDA_CHECK(cudaMalloc(&d_outputBuffer_, outputBufferBytes_));

    // Pinned (page-locked) host memory -- pageable memcpy latency is one
    // of the biggest hidden sources of frame-to-frame jitter; pinned
    // memory gets a direct DMA transfer with much tighter, repeatable
    // timing.
    CUDA_CHECK(cudaHostAlloc(&h_image_, numPixels * sizeof(uint16_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_outputBuffer_, outputBufferBytes_, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_pixel0_, sizeof(PixelStateGPU), cudaHostAllocDefault));
}

void EventGeneratorCUDA::initialize(int width, int height)
{
    freeDeviceBuffers();
    width_ = width;
    height_ = height;
    allocateDeviceBuffers();
    initialized_ = false;
}

void EventGeneratorCUDA::reset()
{
    initialized_ = false;
}

EventPacket EventGeneratorCUDA::generate(const uint16_t *image, int stridePixels, double timestamp)
{
    EventPacket packet{};
    if (image == nullptr || width_ == 0 || height_ == 0)
        return packet;

    // Stage into pinned memory row-by-row (handles stridePixels != width_).
    // If your upstream downscale step can write directly into a pinned
    // buffer, pass that pointer as `image` with stridePixels == width_ and
    // this memcpy collapses to a single cudaMemcpyAsync with no staging.
    for (int y = 0; y < height_; ++y) {
        std::memcpy(h_image_ + static_cast<size_t>(y) * width_,
                    image + static_cast<size_t>(y) * stridePixels,
                    static_cast<size_t>(width_) * sizeof(uint16_t));
    }
    CUDA_CHECK(cudaMemcpyAsync(d_image_, h_image_,
                                static_cast<size_t>(width_) * height_ * sizeof(uint16_t),
                                cudaMemcpyHostToDevice, stream_));

    const dim3 block(32, 8);
    const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);

    if (!initialized_)
    {
        InitFirstFrameKernel<<<grid, block, 0, stream_>>>(
            d_pixelStates_, d_image_, width_, height_, width_, timestamp);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        initialized_ = true;
        packet.startTime = timestamp;
        packet.endTime = timestamp;
        packet.frameNumber = frameNumber_++;
        return packet;
    }

    // Snapshot pixelStates[0].referenceTime BEFORE the kernel updates it --
    // matches the CPU version's `packet.startTime = pixels_[0].referenceTime`.
    // Ordered before the kernel launch on the same stream, so this copy is
    // guaranteed to complete (and read the pre-update value) before the
    // kernel below touches d_pixelStates_[0].
    CUDA_CHECK(cudaMemcpyAsync(h_pixel0_, d_pixelStates_, sizeof(PixelStateGPU),
                                cudaMemcpyDeviceToHost, stream_));

    // Reset only the leading event-count int, not the whole output buffer.
    CUDA_CHECK(cudaMemsetAsync(d_outputBuffer_, 0, sizeof(int), stream_));

    GenerateEventsKernel<<<grid, block, 0, stream_>>>(
        d_pixelStates_, d_image_, width_, height_, width_,
        timestamp, config_.positiveThreshold, config_.negativeThreshold,
        maxEvents_, d_outputBuffer_);
    CUDA_CHECK(cudaGetLastError());

    // Fixed-size D2H copy every single frame (count + full event capacity)
    // -- this is the key change for stable timing. A copy sized to the
    // ACTUAL event count would make transfer time scene-dependent, which
    // is exactly the kind of variability you're trying to remove.
    CUDA_CHECK(cudaMemcpyAsync(h_outputBuffer_, d_outputBuffer_, outputBufferBytes_,
                                cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    int eventCount = 0;
    std::memcpy(&eventCount, h_outputBuffer_, sizeof(int));
    if (eventCount > maxEvents_) {
        fprintf(stderr, "EventGeneratorCUDA: frame produced %d events, capacity is %d "
                "- %d events dropped. Raise maxEventsPerFrame.\n",
                eventCount, maxEvents_, eventCount - maxEvents_);
        eventCount = maxEvents_;
    }

    const Event *devEvents = reinterpret_cast<const Event *>(h_outputBuffer_ + sizeof(int));

    packet.startTime = h_pixel0_->referenceTime;
    packet.endTime = timestamp;
    packet.frameNumber = frameNumber_++;
    packet.events.assign(devEvents, devEvents + eventCount); // O(eventCount), not O(W*H)

    return packet;
}

} // namespace evsim
