#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>
#include <chrono>

#include "ops.h"
#include "manifold.h"

std::tuple<torch::Tensor, torch::Tensor, float>
ComputeWDT(
    const torch::Tensor& positions,
    const torch::Tensor& weights,
    const bool compute_cc) {
    
    int num_points = positions.size(0);
    int dimension = positions.size(1);

    const float* positions_ptr = positions.data_ptr<float>();
    const float* weights_ptr = weights.data_ptr<float>();

    auto dt_result = CGALDDT::WDT(
        num_points, dimension,
        positions_ptr,
        weights_ptr,
        true,
        compute_cc
    );
    
    auto int_options = positions.options().dtype(torch::kInt32);
    auto float_options = positions.options().dtype(torch::kFloat32);
    torch::Tensor dsimp_tensor = torch::zeros({dt_result.num_tri, dimension + 1}, int_options);
    std::memcpy(dsimp_tensor.contiguous().data_ptr<int>(), 
                dt_result.tri_verts_idx, 
                dt_result.num_tri * (dimension + 1) * sizeof(int));
    
    torch::Tensor cc_tensor = torch::zeros({0,}, float_options);
    if (compute_cc) {
        cc_tensor = torch::zeros({dt_result.num_tri, dimension}, float_options);
        std::memcpy(cc_tensor.contiguous().data_ptr<float>(), 
                    dt_result.tri_cc, 
                    dt_result.num_tri * dimension * sizeof(float));
    }
    
    return std::make_tuple(dsimp_tensor, cc_tensor, dt_result.time_sec);
}

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
) {
    return MINDIFFDT::remove_non_manifoldness(
        verts.to(torch::kCPU),
        faces.to(torch::kCPU),
        face_existence.to(torch::kCPU),
        layered_renderer
    );
}