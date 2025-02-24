#ifndef CUDA_MINDIFFDT_MANIFOLD_INCLUDED
#define CUDA_MINDIFFDT_MANIFOLD_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace MINDIFFDT
{
    torch::Tensor remove_non_manifoldness(
        const torch::Tensor& verts,
        const torch::Tensor& faces,
        const torch::Tensor& face_existence,
        py::object& layered_renderer
    );
}

#endif