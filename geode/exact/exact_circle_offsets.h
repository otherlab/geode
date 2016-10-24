#pragma once
#include <geode/exact/circle_quantization.h>
#include <geode/exact/PlanarArcGraph.h>
namespace geode {

Quantized quantize_offset(const Quantizer<Quantized,2>& quant, const real d);

// Adds a curved capsule that is the offset of a single arc (interpreting x0,q,x1 as an ArcSegment) by signed_offset
// Capsule will be wound CCW if signed_offset is positive or CW if signed_offset is negative
// signed_offset must not equal zero
// Note: Added capsule is only approximate. Result may bulge outward/inward by a small number of quantization units (about 3 currently) relative to ideal surface
void add_capsule(ArcAccumulator<Pb::Implicit>& g, const exact::Vec2 x0, const real q, const exact::Vec2 x1, const Quantized signed_offset);
void add_capsule(ArcAccumulator<Pb::Implicit>& g, const ExactArc<Pb::Implicit>& arc, const Quantized signed_offset);

// Given an input shaped defined by a set of closed contours in a planar arc graph, returns a new shape that is grown or shrunk by signed_offset
// If signed offset is positive, this will be all points inside or closer than signed_offset to any point inside the input shape
// If signed offset is negative, this will be all points inside and further than abs(signed_offset) from any point outside of the input shape
// Note: This uses add_capsule which can introduce a small approximation error (see above)
Tuple<Ref<PlanarArcGraph<Pb::Implicit>>, Nested<HalfedgeId>> offset_closed_exact_arcs(const PlanarArcGraph<Pb::Implicit>& src_g, const Nested<HalfedgeId>& contours, const Quantized signed_offset);

} // namespace geode
