#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

namespace py = pybind11;

/*
Intermediate functions used for differentiable Weighted Delaunay Triangulation
*/

// Non differentiable WDT
std::tuple<torch::Tensor, torch::Tensor, float>
ComputeWDT(
    const torch::Tensor& positions,
    const torch::Tensor& weights,
    const bool compute_cc
);

/*
Non-manifoldness
*/
torch::Tensor 
RemoveNonManifold(
    // mesh info
    const torch::Tensor& verts,                         // (V, 3)
    const torch::Tensor& faces,                         // (F, 3)
    const torch::Tensor& face_existence,                // (F)

    // layered renderer
    py::object& layered_renderer
);