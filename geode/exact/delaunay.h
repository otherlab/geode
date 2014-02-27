// Randomized incremental Delaunay using simulation of simplicity
#pragma once

#include <geode/exact/config.h>
#include <geode/mesh/TriangleTopology.h>
namespace geode {

// Approximately Delaunay triangulate a point set, by first quantizing and performing exact Delaunay.
// Any edges are used as constraints in constrained Delaunay.  If two edges intersect, ValueError is thrown.
GEODE_CORE_EXPORT Ref<TriangleTopology> delaunay_points(RawArray<const Vector<real,2>> X,
                                                        RawArray<const Vector<int,2>> edges=Tuple<>(),
                                                        const bool validate=false);

// Exactly Delaunay triangulate a quantized point set.
// Any edges are used as constraints in constrained Delaunay.  If two edges intersect, ValueError is thrown.
GEODE_CORE_EXPORT Ref<TriangleTopology> exact_delaunay_points(RawArray<const Vector<Quantized,2>> X,
                                                              RawArray<const Vector<int,2>> edges=Tuple<>(),
                                                              const bool validate=false);

}
