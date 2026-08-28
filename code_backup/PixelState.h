#pragma once

namespace evsim
{

/**
 * @brief Per-pixel state maintained by the simulator.
 *
 * Every pixel remembers the reference intensity and the
 * timestamp of the last emitted event. This allows
 * multiple events to be generated from a single frame
 * transition, closely matching the behavior of a real DVS.
 */
struct PixelState
{
    /// Reference intensity (raw or log intensity)
    float referenceIntensity = 0.0f;

    /// Timestamp corresponding to referenceIntensity
    double referenceTime = 0.0;

    /// True once the first frame has initialized this pixel
    bool initialized = false;
};

} // namespace evsim