// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <3dgut/splatRaster.h>

#include <3dgut/sensors/cameraModels.h>
// test projection kernel launcher
#include <3dgut/test_projection.h>
// Torch CUDA stream API
#include <ATen/cuda/CUDAContext.h>

threedgut::CameraModelParameters
fromOpenCVPinholeCameraModelParameters(std::array<uint64_t, 2> _resolution,
                                       threedgut::TSensorModel::ShutterType shutter_type,
                                       std::array<float, 2> principal_point,
                                       std::array<float, 2> focal_length,
                                       std::array<float, 6> radial_coeffs,
                                       std::array<float, 2> tangential_coeffs,
                                       std::array<float, 4> thin_prism_coeffs) {
    threedgut::CameraModelParameters params;
    params.shutterType = static_cast<threedgut::TSensorModel::ShutterType>(shutter_type);
    params.modelType   = threedgut::TSensorModel::OpenCVPinholeModel;
    static_assert(sizeof(principal_point) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(focal_length) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(radial_coeffs) == sizeof(tcnn::vec<6>), "[3dgut] typing size mismatch");
    static_assert(sizeof(tangential_coeffs) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(thin_prism_coeffs) == sizeof(tcnn::vec4), "[3dgut] typing size mismatch");
    params.ocvPinholeParams.nearFar          = tcnn::vec2{0.01f, 100.0f};
    params.ocvPinholeParams.principalPoint   = *reinterpret_cast<const tcnn::vec2*>(principal_point.data());
    params.ocvPinholeParams.focalLength      = *reinterpret_cast<const tcnn::vec2*>(focal_length.data());
    params.ocvPinholeParams.radialCoeffs     = *reinterpret_cast<const tcnn::vec<6>*>(radial_coeffs.data());
    params.ocvPinholeParams.tangentialCoeffs = *reinterpret_cast<const tcnn::vec2*>(tangential_coeffs.data());
    params.ocvPinholeParams.thinPrismCoeffs  = *reinterpret_cast<const tcnn::vec4*>(thin_prism_coeffs.data());
    return params;
}

threedgut::CameraModelParameters
fromOpenCVFisheyeCameraModelParameters(std::array<uint64_t, 2> _resolution,
                                       threedgut::TSensorModel::ShutterType shutter_type,
                                       std::array<float, 2> principal_point,
                                       std::array<float, 2> focal_length,
                                       std::array<float, 4> radial_coeffs,
                                       float max_angle) {
    threedgut::CameraModelParameters params;
    params.shutterType = static_cast<threedgut::TSensorModel::ShutterType>(shutter_type);
    params.modelType   = threedgut::TSensorModel::OpenCVFisheyeModel;
    static_assert(sizeof(principal_point) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(focal_length) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(radial_coeffs) == sizeof(tcnn::vec4), "[3dgut] typing size mismatch");
    params.ocvFisheyeParams.principalPoint = *reinterpret_cast<const tcnn::vec2*>(principal_point.data());
    params.ocvFisheyeParams.focalLength    = *reinterpret_cast<const tcnn::vec2*>(focal_length.data());
    params.ocvFisheyeParams.radialCoeffs   = *reinterpret_cast<const tcnn::vec4*>(radial_coeffs.data());
    params.ocvFisheyeParams.maxAngle       = max_angle;
    return params;
}

threedgut::CameraModelParameters
fromEquirectangularCameraModelParameters(std::array<uint64_t, 2> /*_resolution*/,
                                        threedgut::TSensorModel::ShutterType shutter_type) {
    threedgut::CameraModelParameters params;
    params.shutterType = static_cast<threedgut::TSensorModel::ShutterType>(shutter_type);
    params.modelType   = threedgut::TSensorModel::EquirectangularModel;
    // No additional parameters for initial ERP support
    return params;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    pybind11::class_<SplatRaster>(m, "SplatRaster")
        .def(pybind11::init<const nlohmann::json&>())
        .def("trace", &SplatRaster::trace)
        .def("trace_bwd", &SplatRaster::traceBwd)
        .def("collect_times", &SplatRaster::collectTimes);

    py::enum_<threedgut::TSensorModel::ShutterType>(m, "ShutterType")
        .value("ROLLING_TOP_TO_BOTTOM", threedgut::TSensorModel::ShutterType::RollingTopToBottomShutter)
        .value("ROLLING_LEFT_TO_RIGHT", threedgut::TSensorModel::ShutterType::RollingLeftToRightShutter)
        .value("ROLLING_BOTTOM_TO_TOP", threedgut::TSensorModel::ShutterType::RollingBottomToTopShutter)
        .value("ROLLING_RIGHT_TO_LEFT", threedgut::TSensorModel::ShutterType::RollingRightToLeftShutter)
        .value("GLOBAL", threedgut::TSensorModel::ShutterType::GlobalShutter);

    py::class_<threedgut::CameraModelParameters>(m, "CameraModelParameters")
        .def(py::init<>());

    m.def("fromOpenCVPinholeCameraModelParameters", &fromOpenCVPinholeCameraModelParameters,
          py::arg("resolution"),
          py::arg("shutter_type"),
          py::arg("principal_point"),
          py::arg("focal_length"),
          py::arg("radial_coeffs"),
          py::arg("tangential_coeffs"),
          py::arg("thin_prism_coeffs"));

    m.def("fromOpenCVFisheyeCameraModelParameters", &fromOpenCVFisheyeCameraModelParameters,
          py::arg("resolution"),
          py::arg("shutter_type"),
          py::arg("principal_point"),
          py::arg("focal_length"),
          py::arg("radial_coeffs"),
          py::arg("max_angle"));

    m.def("fromEquirectangularCameraModelParameters", &fromEquirectangularCameraModelParameters,
          py::arg("resolution"),
          py::arg("shutter_type"));

    // Lightweight test entrypoint: project batch of 3D points with given model
    m.def(
        "_test_project_points",
        [](const threedgut::CameraModelParameters& params,
           std::array<uint64_t, 2> resolution,
           torch::Tensor positions) {
            TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
            TORCH_CHECK(positions.dtype() == torch::kFloat32, "positions must be float32");
            TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3, "positions must be [N,3]");
            auto N = static_cast<int>(positions.size(0));

            auto options_f = positions.options().dtype(torch::kFloat32);
            auto options_b = positions.options().dtype(torch::kUInt8);
            torch::Tensor projected = torch::empty({N, 2}, options_f);
            torch::Tensor valids    = torch::empty({N}, options_b);

            // Launch on current stream
            auto stream = at::cuda::getCurrentCUDAStream();

            project_points_cuda(
                params,
                tcnn::ivec2{static_cast<int>(resolution[0]), static_cast<int>(resolution[1])},
                reinterpret_cast<const tcnn::vec3*>(positions.data_ptr<float>()),
                reinterpret_cast<tcnn::vec2*>(projected.data_ptr<float>()),
                reinterpret_cast<uint8_t*>(valids.data_ptr<uint8_t>()),
                N,
                stream.stream());

            return py::make_tuple(projected, valids);
        },
        py::arg("params"),
        py::arg("resolution"),
        py::arg("positions"));
}
