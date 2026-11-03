#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// src: pitch-linear uint16 device image; dst: tightly packed uint16 (dw*dh).
// Synchronous: returns after the kernel has finished.
cudaError_t launchResizeRawU16(const uint16_t *src, size_t srcPitchBytes, int sw, int sh,
                               uint16_t *dst, int dw, int dh);