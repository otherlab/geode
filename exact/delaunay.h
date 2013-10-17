// Randomized incremental Delaunay using simulation of simplicity
#pragma once

#include <other/core/exact/config.h>
#include <other/core/mesh/CornerMesh.h>
namespace other {

// Compute an approximate Delaunay triangulation of a point set, by first quantizing and performing exact Delaunay.
OTHER_CORE_EXPORT Ref<CornerMesh> delaunay_points(RawArray<const Vector<real,2>> X, bool validate=false);

// Compute the exact Delaunay triangulation of a quantized point set
OTHER_CORE_EXPORT Ref<CornerMesh> exact_delaunay_points(RawArray<const Vector<Quantized,2>> X, bool validate=false);

}
