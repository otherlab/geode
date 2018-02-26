#include "simplify_arcs.h"

#include <geode/array/sort.h>
#include <geode/exact/circle_csg.h>
#include <geode/geometry/arc_fitting.h>

#include <queue>
namespace geode {

namespace {
struct ArcVertex {
  ArcVertex* prev;
  ArcVertex* next;
  Vec2 pos;
  real q_to_next;
  real slack_to_next;
  int change_marker; // Track any changes to segment between pos and next
  bool is_erased() const { return change_marker == -1; }
};

struct CollapseOperation
{
  Vector<ArcVertex*, 3> verts;
  Vec3i change_markers;
  real new_q;
  real error_cost;
  static Vec3i get_change_markers(Vector<ArcVertex*, 3> verts) { return Vec3i(verts[0]->change_marker, verts[1]->change_marker, verts[2]->change_marker); }
  bool operator<(const CollapseOperation& rhs) const { return error_cost > rhs.error_cost; }
  CollapseOperation(ArcVertex* v, real _new_q, real _error_cost)
  : verts(v->prev, v, v->next)
  , change_markers(get_change_markers(verts))
  , new_q(_new_q)
  , error_cost(_error_cost)
  {
    assert(!change_markers.contains(-1)); // Shouldn't have any erased verts
  }
  bool still_valid() const { return change_markers == get_change_markers(verts); }
};

struct SimplifyVertResult {
  SimplifyVertResult(const real _new_q, const real _error_cost) : new_q(_new_q), error_cost(_error_cost) {}
  real new_q;
  real error_cost;
};
} // anonymous namespace

#if 0
static real safe_avg(const real x0, const real w0, const real x1, const real w1) {
  const auto dw = w1 - w0;
  const real result = (x0*w0 + x1*w1) / (w1 + w0);
  if(isnan(result))
    return (x0 + x1) / 2;
  return clamp(result, min(x0,x1), max(x0,x1));
}

// q = cl / (1 + sqrt(1 - sqr(cl)))
static real fit_q(const Vec2 x0, const real q01, const Vec2 x1, const real q12, const Vec2 x2) {
  const auto a01 = ArcSegment(x0,x1,q01), a12 = ArcSegment(x1,x2,q12);
  const auto avg_c = safe_avg(a01.c(), a01.arc_length(), a12.c(), a12.arc_length());
  const auto new_l = (x1 - x0).magnitude();
  const auto new_cl = avg_c * new_l;

}
#endif

// result.error_bound will be equal to inf for unhandled corner cases
static SimplifyVertResult colinear_arc_fit(const Vec2 x0, const real q01, const Vec2 x1, const real q12, const Vec2 x2) {
  if(abs(q01) >= 1 || abs(q12) >= 1) return SimplifyVertResult(0,inf);

  const real q02 = fit_q(x0,x2,x1); // This is q value for arc from x0 to x2 that goes through x1

  // Use !(abs(q) < 1) so that nans won't pass
  if(!(abs(q02) < 1)) return SimplifyVertResult(0,inf);

  const real l02_sqr = 0.25*(x2 - x0).sqr_magnitude();
  const real l02 = sqrt(l02_sqr);
  const real sin_02 = 2*q02 / (1 + sqr(q02));

  const real c = sin_02 / l02; // Signed curvature of new arc
  const real c_sqr = sqr(sin_02) / l02_sqr;

  const real l01_sqr = 0.5*(x1-x0).sqr_magnitude();
  const real l12_sqr = 0.5*(x2-x1).sqr_magnitude();
  const real l01 = sqrt(l01_sqr);
  const real l12 = sqrt(l12_sqr);
  const real new_q01 = c*l01 / (1 + sqrt(1 - c_sqr*l01_sqr));
  const real new_q12 = c*l12 / (1 + sqrt(1 - c_sqr*l12_sqr));

  // Use !(abs(q) < 1) so that nans won't pass (can happen if lengths are close to zero)
  if(!(abs(new_q01) < 1) || !(abs(new_q12) < 1)) return SimplifyVertResult(0,inf);

  const real error_01 = abs((q01 - new_q01)*l01);
  const real error_12 = abs((q12 - new_q12)*l12);

  return SimplifyVertResult(q02,max(abs(error_01),abs(error_12)));
}

// Error bound for replacing two consecutive arcs from x0 to x1 to x2 with single arc from x0 to x2 keeping the better of the two q value
// result.error_bound will be equal to inf for unhandled corner cases
static SimplifyVertResult erase_vertex_error(const Vec2 x0, const real q01, const Vec2 x1, const real q12, const Vec2 x2) {
  // If we erase this vertex, we move x1 to x2
  // If we change the position of a vertex by delta, maximum movement of any point on a connected arc segments <= delta.magnitude()*(1+abs(q))
  // Probably could work out a tighter bound on this error, but it should be sufficient to collapse repeated points
  auto best = SimplifyVertResult(0, inf); // Default to inf
  // Check collapsing x1 to either x0 or x2
  for(int dir = 0; dir < 2; ++dir)
  {
    const real clobbered_q = dir ? q01 : q12;
    if(abs(clobbered_q) < 1) {  // Only try to collapse arcs that are less than a half circle
      const real kept_q = dir ? q12 : q01;
      const Vec2 new_pos_for_x1 = dir ? x0 : x2;
      const real mag_delta = (new_pos_for_x1 - x1).magnitude(); // Distance collapsed point moves
      const real clobbered_arc_error = mag_delta; // Since abs(clobbered_q) <= 1, collapsed arc is less than a half circle and no point is further from new point than mag_delta
      const real moved_arc_error = (1 + abs(kept_q))*mag_delta; // Movement of points on kept arc can be magnified by kept_q

      const real new_error = max(clobbered_arc_error, moved_arc_error);
      if(new_error < best.error_cost) {
        best = SimplifyVertResult(kept_q, new_error);
      }
    }
  }
  return best;
}

typedef std::priority_queue<CollapseOperation, std::vector<CollapseOperation>, std::less<CollapseOperation> > CollapseQueue;

static void enqueue_checks(ArcVertex* v, CollapseQueue& queue) {
  if(!v->next || !v->prev) return; // Can't collapse either end vertex
  if(v->next == v->prev) return; // Can't collapse into one vertex
  assert(v->next != v && v->prev != v); // Shouldn't be connected to self (except if v->next == v->prev which is caught above)

  // Walk pointers to get actual arc data
  const Vec2 x0 = v->prev->pos;
  const real q01 = v->prev->q_to_next;
  const Vec2 x1 = v->pos;
  const real q12 = v->q_to_next;
  const Vec2 x2 = v->next->pos;

  // Compare operations
  const SimplifyVertResult colinear_op = colinear_arc_fit(x0, q01, x1, q12, x2);
  const SimplifyVertResult erase_op = erase_vertex_error(x0, q01, x1, q12, x2);
  const SimplifyVertResult& best_op = (colinear_op.error_cost <= erase_op.error_cost) ? colinear_op : erase_op;
  assert(best_op.error_cost >= 0);
  assert(!v->prev->is_erased() && !v->is_erased() && !v->next->is_erased());
  // Stick operation into queue
  if(best_op.error_cost < min(v->prev->slack_to_next, v->slack_to_next)) {
    queue.push(CollapseOperation(v, best_op.new_q, best_op.error_cost));
  }
}

Array<CircleArc> simplify_arcs(const RawArray<const CircleArc> input, const real max_allowed_change, const bool is_closed) {
  const int num_verts = input.size();
  if(num_verts <= 1) return input.copy();
  Array<ArcVertex> data(num_verts);
  
  for(int i : range(num_verts)) {
    ArcVertex& v = data[i];
    v.prev = (i - 1 >=        0) ? (&data[i-1]) : (is_closed ? &data.back() : 0);
    v.next = (i + 1 < num_verts) ? (&data[i+1]) : (is_closed ? &data.front() : 0);
    v.pos = input[i].x;
    v.q_to_next = input[i].q;
    v.slack_to_next = max_allowed_change;
    v.change_marker = 0;
    GEODE_ASSERT(!v.is_erased());
  }

  CollapseQueue queue;
  for(int i : range(num_verts)) {
    enqueue_checks(&data[i], queue);
  }

  while(!queue.empty()) {
    const CollapseOperation op = queue.top();
    queue.pop();
  
    if(op.still_valid()) {
      const real old_slack = min(op.verts[0]->slack_to_next, op.verts[1]->slack_to_next);
      const real new_slack = old_slack - op.error_cost;
      if(new_slack >= 0) {
        ArcVertex *prev = op.verts[0], *curr = op.verts[1], *next = op.verts[2];
        assert(curr && prev && next);
        assert(!prev->is_erased() && !curr->is_erased() && !next->is_erased());
        prev->q_to_next = op.new_q; // Update q value for previous segment
        ++prev->change_marker; // Track that previous segment was changed so we recompute any ops from the queue

        assert(new_slack <= prev->slack_to_next); // Should always have decreasing slack
        prev->slack_to_next = new_slack; // Update slack

        prev->next = next; // Link prev and next with each other
        next->prev = prev;
        curr->change_marker = -1; // Mark curr as erased
        
        enqueue_checks(prev, queue); // Recheck prev/next with new states 
        enqueue_checks(next, queue);
      }
    }
  }

  Array<CircleArc> result;
  for(const auto& info : data) {
    if(!info.is_erased()) {
      result.append(CircleArc(info.pos,info.q_to_next));
    }
  }
  return result;
}

Nested<CircleArc> simplify_arcs(const Nested<const CircleArc> input, const real max_point_movement, const bool is_closed) {
  Nested<CircleArc, false> result;
  for(const auto& polyarc : input) {
    result.append(simplify_arcs(polyarc, max_point_movement, is_closed));
  }
  return result.freeze();
}

} // namespace geode
