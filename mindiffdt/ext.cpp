#include <torch/extension.h>
#include "mindiffdt.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // compute weighted delaunay triangulation
    m.def("compute_wdt", &ComputeWDT);

    // non-manifoldness
    m.def("remove_non_manifold", &RemoveNonManifold);
}