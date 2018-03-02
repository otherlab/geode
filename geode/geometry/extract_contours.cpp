#include "extract_contours.h"
#include <geode/array/Array2d.h>

namespace geode {

namespace {
struct XYIter {
  Vec2i max;
  Vec2i i;
  explicit XYIter(Vec2i _max, Vec2i _i) : max(_max), i(_i) {}
  bool operator!=(XYIter j) const { return i != j.i; } // Assumes sizes are equal
  XYIter& operator++() { ++i.x; if(i.x == max.x) { i.x = 0; ++i.y; } return *this; }
  XYIter& operator--() { if(i.x == 0) { i.x = max.x; --i.y; } --i.x; return *this; }
  Vec2i operator*() const { return i; }
};
static Range<XYIter> xy_range(const Vec2i sizes) { return range(XYIter(sizes,Vec2i()),XYIter(sizes,Vec2i(0,sizes.y))); }

typedef unsigned char FlagsData;

template<FlagsData _set, FlagsData _cleared> struct FlagSubsetCondition {
  static const FlagsData set_flags = _set;
  static const FlagsData cleared_flags = _cleared;
  static const FlagsData used_flags = set_flags | cleared_flags;
  static_assert((set_flags & cleared_flags) == 0, "Overlap between set and cleared flags");

  // This returns a FlagSubsetCondition that checks the logical and of two other conditions
  template<typename RhsType> FlagSubsetCondition<set_flags | RhsType::set_flags, cleared_flags | RhsType::cleared_flags> operator&&(const RhsType rhs) const {
    return FlagSubsetCondition<set_flags | RhsType::set_flags, cleared_flags | RhsType::cleared_flags>();
  }
  static bool check(const FlagsData data) { return (data & used_flags) == set_flags; }
  explicit operator bool() const = delete; // Conditions shouldn't get converted to bools
};

template<int bit> struct Flag {
  static_assert(bit < 8*sizeof(FlagsData), "Bit index doesn't fit in FlagsData");
  static constexpr FlagsData mask = (1<<bit);
  static constexpr FlagsData t = mask; // True bits mask
  static constexpr FlagsData f = 0; // False bits mask
  typedef FlagSubsetCondition<mask,0> is_true;
  typedef FlagSubsetCondition<0,mask> is_false;
};

enum class GridDir { E, N, W, S };

static Vec2i dir_offset(const GridDir d) {
  switch(d) {
    case GridDir::E: return Vec2i( 1, 0);
    case GridDir::N: return Vec2i( 0, 1);
    case GridDir::W: return Vec2i(-1, 0);
    case GridDir::S: return Vec2i( 0,-1);
  }
  GEODE_FATAL_ERROR("Unhandled case in switch statement");
}

static int index_of(const GridDir dir) {
  static_assert(static_cast<int>(GridDir::E) == 0, "Enum constant mismatch for GridDir::E");
  static_assert(static_cast<int>(GridDir::N) == 1, "Enum constant mismatch for GridDir::N");
  static_assert(static_cast<int>(GridDir::W) == 2, "Enum constant mismatch for GridDir::W");
  static_assert(static_cast<int>(GridDir::S) == 3, "Enum constant mismatch for GridDir::S");
  return static_cast<int>(dir);
}

struct GridVertex {
  enum VertexFlagIndex { ne_filled_bit, nw_filled_bit, sw_filled_bit, se_filled_bit, mark_bit};
  typedef Flag<ne_filled_bit> ne;
  typedef Flag<nw_filled_bit> nw;
  typedef Flag<sw_filled_bit> sw;
  typedef Flag<se_filled_bit> se;
  typedef Flag<mark_bit> marked;
  typedef decltype(ne::is_true() && se::is_false() && marked::is_false()) unmarked_e_edge;

  FlagsData info;

  void mark() { info |= marked::mask; }
  bool is_marked() const { return marked::is_true::check(info); }
  bool is_unmarked_e_edge() const { return unmarked_e_edge::check(info); }


  GridDir next_dir(const GridDir prev_dir) const {
    const FlagsData neighbors_mask = ne::mask | nw::mask | sw::mask | se::mask;
    const GridDir ERROR_DIR = GridDir::W; // Fallback value to return if we are in a bad state
    assert((info & neighbors_mask) != (se::f | sw::f | nw::f | ne::f));
    assert((info & neighbors_mask) != (se::t | sw::t | nw::t | ne::t));

    switch(info & neighbors_mask) {
      case (se::f | sw::f | nw::f | ne::f): return ERROR_DIR; // Attempted to traverse empty space.
      case (se::f | sw::f | nw::f | ne::t): return GridDir::E;
      case (se::f | sw::f | nw::t | ne::f): return GridDir::N;
      case (se::f | sw::f | nw::t | ne::t): return GridDir::E;

      case (se::f | sw::t | nw::f | ne::f): return GridDir::W;
      case (se::f | sw::t | nw::f | ne::t): return prev_dir == GridDir::S ? GridDir::W : GridDir::E;
      case (se::f | sw::t | nw::t | ne::f): return GridDir::N;
      case (se::f | sw::t | nw::t | ne::t): return GridDir::E;

      case (se::t | sw::f | nw::f | ne::f): return GridDir::S;
      case (se::t | sw::f | nw::f | ne::t): return GridDir::S;
      case (se::t | sw::f | nw::t | ne::f): return prev_dir == GridDir::E ? GridDir::S : GridDir::N;
      case (se::t | sw::f | nw::t | ne::t): return GridDir::S;

      case (se::t | sw::t | nw::f | ne::f): return GridDir::W;
      case (se::t | sw::t | nw::f | ne::t): return GridDir::W;
      case (se::t | sw::t | nw::t | ne::f): return GridDir::N;
      case (se::t | sw::t | nw::t | ne::t): return ERROR_DIR; // Attempted to traverse solid space
    }
    GEODE_FATAL_ERROR("Unhandled case in switch statement");
  }
};

} // end anonymous namespace

static Vec2 interpolate_vertex(const Vec2i vert_index, const GridDir dir, const RawArray<const real, 2> samples, const real contour_edge_threshold, const real default_sample_value) {
  // 'vert_index' points to corner between 4 samples. 'vert_index + sample_offsets' are indicies into 'samples' for adjacent samples
  static const Vector<Vec2i,4> sample_offsets = vec(Vec2i( 0, 0), Vec2i(-1, 0), Vec2i(-1,-1), Vec2i( 0,-1));
  const Vec2i il = vert_index + sample_offsets[(index_of(dir) + 0) % 4]; // Index of sample on the left of ray from 'vert_index' pointing in 'dir'
  const Vec2i ir = vert_index + sample_offsets[(index_of(dir) + 3) % 4]; // Index of sample on the right of ray from 'vert_index' pointing in 'dir'
  const real sr = samples.valid(ir) ? samples[ir] : default_sample_value;
  const real sl = samples.valid(il) ? samples[il] : default_sample_value;
  const real t = clamp((contour_edge_threshold - 0.5*(sr + sl)) / (sl - sr), -0.5, 0.5); // Solve for 'contour_edge_threshold' in interpolation between 'sr' and 'sl'
  const Vec2 dir_vec = Vec2(dir_offset(dir));
  return Vec2(vert_index) + Vec2(-0.5,-0.5) + // Start at center of the 4 adjacent samples and add...
          0.5*dir_vec + // ...to get middle of line between 'sr' and 'sl' then add...
          t*rotate_left_90(dir_vec); // ...to get interpolated point (since range of t is [-0.5,0.5])
}

Nested<Vec2> extract_contours(const RawArray<const real, 2> samples, const real contour_edge_threshold, const real default_sample_value) {
  if(samples.total_size() == 0)
    return Nested<Vec2>();
  const Vec2i src_sizes = samples.sizes();

  // Copy samples into a buffer that includes local neighborhood and other info
  const auto verts = Array<GridVertex, 2>(src_sizes + Vec2i::ones()); // expand size by 1 so we can represent all edges

  for(const Vec2i src_idx : xy_range(src_sizes)) {
    if(samples[src_idx] > contour_edge_threshold) { // Write each interior sample in source to the 4 adjacent vertices
      verts[src_idx + Vec2i(0,0)].info |= GridVertex::ne::mask;
      verts[src_idx + Vec2i(1,0)].info |= GridVertex::nw::mask;
      verts[src_idx + Vec2i(1,1)].info |= GridVertex::sw::mask;
      verts[src_idx + Vec2i(0,1)].info |= GridVertex::se::mask;
    }
  }

  Nested<Vec2, false> result;
  for(const Vec2i seed : xy_range(verts.sizes())) {
    if(verts[seed].is_unmarked_e_edge()) {
      // We have found an unmarked vertex that has an edge to the east
      result.append_empty(); // Start a new contour
      GridDir dir = GridDir::E; // Force direction to E. Important if we are on a 'saddle' vertex with two opposite filled corners
      Vec2i curr_index = seed;
      for(;;) {
        if(dir == GridDir::E) { // We only mark eastward edges
          auto& vert = verts[curr_index];
          if(vert.is_marked()) // If we already marked this one we are done
            break;
          vert.mark(); // Mark edge so that we won't reuse it as a seed and will stop when we loop back
        }
        result.append_to_back(interpolate_vertex(curr_index, dir, samples, contour_edge_threshold, default_sample_value));
        assert(result.back().size() <= 4*samples.total_size()); // Check that we aren't caught in an infinite loop
        curr_index += dir_offset(dir); // Walk in given direction
        dir = verts[curr_index].next_dir(dir);
      }
    }
  }
  return result.freeze();
}

} // namespace geode
