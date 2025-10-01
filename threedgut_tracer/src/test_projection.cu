// Test kernel to exercise cameraProjections for different models.

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <cuda_runtime.h>

#include <3dgut/kernels/cuda/sensors/cameraProjections.cuh>
#include <3dgut/test_projection.h>

using namespace tcnn;
using namespace threedgut;

namespace {

__global__ void project_kernel(CameraModelParameters sensorModel,
                               ivec2 resolution,
                               const vec3* __restrict__ positions,
                               vec2* __restrict__ projected,
                               uint8_t* __restrict__ valids,
                               int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    vec2 p;
    const bool ok = projectPoint(sensorModel, resolution, positions[idx], 0.0f, p);
    projected[idx] = p;
    valids[idx]    = static_cast<uint8_t>(ok ? 1 : 0);
}

} // namespace

void project_points_cuda(const CameraModelParameters& sensorModel,
                         const ivec2& resolution,
                         const vec3* positions,
                         vec2* projected,
                         uint8_t* valids,
                         int N,
                         cudaStream_t stream) {
    constexpr int kBlock = 256;
    int grid             = (N + kBlock - 1) / kBlock;
    project_kernel<<<grid, kBlock, 0, stream>>>(sensorModel, resolution, positions, projected, valids, N);
}

