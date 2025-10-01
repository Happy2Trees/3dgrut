// Minimal test-only projection kernel launcher
#pragma once

#include <cuda_runtime.h>

#include <3dgut/sensors/cameraModels.h>
#include <tiny-cuda-nn/vec.h>

// Launches a CUDA kernel to project N positions with given sensor model.
// positions, projected live on device; valids written as 0/1 bytes.
void project_points_cuda(const threedgut::CameraModelParameters& sensorModel,
                         const tcnn::ivec2& resolution,
                         const tcnn::vec3* positions,
                         tcnn::vec2* projected,
                         uint8_t* valids,
                         int N,
                         cudaStream_t stream);

