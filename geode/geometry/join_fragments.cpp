#include "join_fragments.h"

#include <geode/math/constants.h>
#include <geode/geometry/ParticleTree.h>
#include <geode/structure/Tuple.h>
#include <queue>

namespace geode {

namespace {
struct Match {
  Match() : start(-1), end(-1), dist(inf) {}
  Match(int _start, int _end, real _dist) : start(_start), end(_end), dist(_dist) {}
  int start, end;
  real dist;
  void reset() { end = -1;}
  bool is_set() const { return end != -1; }
  bool operator<(const Match& rhs) const { return dist >= rhs.dist; } // Reverse ordering so that greatest element will have smallest distance
};
}

static void closest_match_helper(const ParticleTree<Vec2>& tree, const Array<Match>& current_matches, const Vec2& point, const int point_id, int& match_id, real& sqr_distance, const int node) {
  if (!tree.is_leaf(node)) {
    Vector<real,2> bounds(tree.boxes[2*node+1].sqr_distance_bound(point),
                          tree.boxes[2*node+2].sqr_distance_bound(point));
    const int c = bounds.argmin();
    if (bounds[c]<sqr_distance)
      closest_match_helper(tree,current_matches,point,point_id,match_id,sqr_distance,2*node+1+c);
    if (bounds[1-c]<sqr_distance)
      closest_match_helper(tree,current_matches,point,point_id,match_id,sqr_distance,2*node+2-c);
  }
  else {
    for (int p : tree.prims(node)) {
      // This check sets what matches are excluded
      // Currently we don't match a point to itself or a previously matched point
      if(p == point_id || current_matches[p].is_set()) continue;

      real new_sqr_d = sqr_magnitude(point-tree.X[p]);
      if (new_sqr_d < sqr_distance) {
        sqr_distance = new_sqr_d;
        match_id = p;
      }
    }
  }
}

static Tuple<int, real> closest_match(const ParticleTree<Vec2>& tree, const Array<Match>& current_matches, const int point_id) {
  int match_id = -1;
  real sqr_distance = inf;
  if(tree.nodes()) {
    closest_match_helper(tree, current_matches, tree.X[point_id], point_id, match_id, sqr_distance, 0);
  }
  return tuple(match_id, sqrt(sqr_distance));
}

static Vec2 get_endpoint(const Vec2& p) {
  return p;
}

static Vec2 get_endpoint(const CircleArc& p) {
  return p.x;
}

// This depends on the logic in join_fragments to update the offset array when we complete a fragment
static void add_fragment(Nested<Vec2, false>& result, const RawArray<Vec2>& fragment, bool reversed) {
  const int fragment_start = result.flat.size();
  result.flat.extend(fragment); // Add the points inside this fragment
  if(reversed) result.flat.slice(fragment_start, result.flat.size()).reverse(); // Reverse new points if we are walking backwards
}

// This depends on the logic in join_fragments to update the offset array when we complete a fragment
static void add_fragment(Nested<CircleArc, false>& result, const RawArray<CircleArc>& fragment, bool reversed) {
  const int fragment_start = result.flat.size();
  result.flat.extend(fragment); // Add the points inside this fragment
  if(reversed) reverse_arcs(result.flat.slice(fragment_start, result.flat.size()));
  // This just inserts straight lines between unconnected points; other options like smoothing could be done if needed
  result.flat.back().q = 0;
}

Array<Match> match_endpoints(const Array<Vec2> &X) {

  // Build a tree to search for closest matches
  Ref<ParticleTree<Vec2>> tree = new_<ParticleTree<Vec2>>(X,8);

  // Build initial set of matches
  Array<Match> current_matches(X.size());
  for(int i : range(X.size())) {
    current_matches[i].start = i;
    current_matches[i].end = -1;
    current_matches[i].dist = inf;
  }

  // Initilize a queue with closest match for each endpoint
  // This queue should always have the closest possible match for all unmatched points
  // Some matches in the queue might become invalid if the opposite end gets matched
  std::priority_queue<Match> match_queue;
  for(int i : range(X.size())) {
    const Tuple<int, real> new_match = closest_match(tree, current_matches, i);
    if(new_match.x == -1) continue;
    match_queue.push(Match(i,new_match.x,new_match.y));
  }

  // Process queue until everything gets matched
  while(!match_queue.empty()) {
    const Match next = match_queue.top();
    match_queue.pop();

    Match& start_match = current_matches[next.start];
    Match& end_match = current_matches[next.end];

    assert(start_match.dist >= next.dist); // We should be checking matches in increasing distances

    if(!start_match.is_set() && !end_match.is_set()) {
      // Mark match at start and end
      start_match.end = end_match.start;
      end_match.end = start_match.start;
      start_match.dist = next.dist;
      end_match.dist = next.dist;
    }
    else {
      if(start_match.is_set()) {
        // Matches are marked from best possible to worst. If we are already matched, the other match should be as good or better
        // The only way this ought to happen is if...
        assert(Vec2i(start_match.start, start_match.end).sorted() != Vec2i(next.start, next.end).sorted() // ... we found the same match from the opposite side...
          || (start_match.dist == next.dist)); // ...or we have multiple matches that are exactly the same distance (usually three segments that end at the same point)

        // Regardless, this point already has a match so we don't need to queue anything else
      }
      else {
        assert(!start_match.is_set() && end_match.is_set()); // Should only reach here when start is unmatched and end is matched
        assert(end_match.dist <= next.dist); // Opposite end should have found something better

        // Find a new candidate for the start
        const Tuple<int,real> new_closest = closest_match(tree, current_matches, next.start);
        assert(new_closest.x != -1); // Check that we found a new match
        assert(new_closest.x != next.end); // Check that we didn't find the same match again
        assert(new_closest.y >= next.dist); // We should only be finding worse candidates
        // Queue up the new match to be checked
        match_queue.push(Match(next.start, new_closest.x, new_closest.y));
      }
    }
  }

  return current_matches;
}

// Join endpoints of fragments to try and make closed contours
// Joins are made greedily starting with the closest unjoined pair of endpoints
// No maximum distance is currently used for joining fragments
// Fragments are assumed to be unoriented
// Fragments are allowed to connect back to themselves (modify closest_match_helper to change this)
template<class T>
Nested<T> join_fragments_helper(const Nested<T>& fragments) {

  // Build array of endpoints
  Array<Vec2> X;
  for(const auto& f : fragments) {
    assert(!f.empty());
    X.append(get_endpoint(f.front()));
    X.append(get_endpoint(f.back()));
  }

  Array<Match> current_matches = match_endpoints(X);

  // At this point each element in current_matches should have the join to the next fragment
  // We need to walk these matches and build a single nested array for the result
  // We build fragments by directly appending to the flat array, and update the offset array once the fragment is complete
  Nested<T,false> result;
  for(const int seed_match : range(current_matches.size())) {
    int curr = seed_match;
    while(current_matches[curr].is_set()) {
      const bool reversed = (curr & 1) != 0; // Even ids are starts of fragments, odd ids are ends of fragments
      add_fragment(result, fragments[curr>>1], reversed);
      const int next = current_matches[curr^1].end; // Next fragment comes at opposite end of the current fragment
      current_matches[curr].reset(); // Reset matches for this fragment to mark it as traversed
      current_matches[curr^1].reset();
      curr = next;
    }
    // Add new offset for fragment (assuming we added anything)
    if(result.offsets.back() != result.flat.size()) {
      result.offsets.append(result.flat.size());
    }
  }

  return result.freeze();
}

Nested<Vec2> join_fragments(const Nested<Vec2>& fragments) {
  return join_fragments_helper(fragments);
}

Nested<CircleArc> join_fragments(const Nested<CircleArc>& fragments) {
  return join_fragments_helper(fragments);
}

} // namespace geode
