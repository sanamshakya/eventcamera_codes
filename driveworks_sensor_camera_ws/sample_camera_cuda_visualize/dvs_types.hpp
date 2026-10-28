// dvs_types.hpp
// Core data types for the C++ port of the DVS-Voltmeter event simulator.
// Ported from: config.py, simulator.py (Python/NumPy/PyTorch reference implementation)

#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace dvs {

// One DVS event: timestamp in microseconds, pixel coords, polarity (1 = ON, 0 = OFF).
// Mirrors the [t, x, y, p] row layout used throughout the Python reference (simulator.py,
// visualize.py, event_filter.py).
struct Event {
    int64_t  t;   // timestamp, microseconds
    int32_t  x;
    int32_t  y;
    uint8_t  p;   // 1 = ON (positive), 0 = OFF (negative)
};

// Sensor noise-model coefficients k1..k6, equivalent to cfg.SENSOR.K in config.py.
// Meaning (per the DVS-Voltmeter paper this code implements):
//   k1: contrast-sensitivity gain on the drift term
//   k2: photocurrent offset in the drift denominator
//   k3: diffusion (shot-noise) coefficient
//   k4: constant drift (dark current) term
//   k5: illumination-dependent drift term
//   k6: constant diffusion floor
struct SensorK {
    double k1 = 0.0;
    double k2 = 0.0;
    double k3 = 0.0;
    double k4 = 0.0;
    double k5 = 0.0;
    double k6 = 0.0;
};

// Convenience presets matching the branches in config.py. Values copied verbatim.
inline SensorK sensor_k_preset(const std::string& camera_type) {
    if (camera_type == "DVS346")
        return SensorK{0.00018 * 29250, 20, 0.0001, 1e-7, 5e-9, 0.00001};
    if (camera_type == "DVS240")
        return SensorK{0.000094 * 47065, 23, 0.0002, 1e-7, 5e-8, 0.00001};
    if (camera_type == "Raw2DVS346")
        return SensorK{2.388, 4.166e-7, 1.541e-6, 9.768e-8, 1.466e-11, 9.824e-6};
    if (camera_type == "RGB2DVS346")
        return SensorK{5.332474147628972, 0.9003332027266823, 8.288263352543993e-06,
                        1.0992397172828087e-07, 3.302652963977818e-09, 1.1012444038716504e-07};
    return SensorK{}; // caller should validate camera_type before relying on this
}

// A plain, contiguous grayscale frame buffer (row-major, height x width).
// This is intentionally decoupled from any specific camera SDK type (cv::Mat,
// NvSciBufObj-mapped pointer, etc.) — see camera_frame_source.hpp for the adapter layer
// that bridges a real capture pipeline (e.g. NVSIPL) into this type.
struct FrameView {
    const double* data = nullptr; // row-major, size = width*height, intensity in same units as k1..k6 expect
    int width = 0;
    int height = 0;
};

} // namespace dvs
