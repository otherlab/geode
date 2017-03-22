#include "RankedTree.h"
#include <geode/array/Array.h>
#include <geode/math/integer_log.h>
#include <geode/python/wrap.h>
#include <geode/random/permute.h>

namespace geode {


static auto random_permutation(const int length, const int seed) -> Array<int> {
  uint128_t key = 12345 + seed;
  const auto result = Array<int>{length, uninit};
  for(const int i : range(length)) {
    result[i] = static_cast<int>(random_permute(length, key, i));
  }
  return result;
}

// Add a bunch of integers to a tree in order making sure that that is what tree contains at every step
static void try_integer_range() {
  for(const int seed : {0,1,2,3}) {
    for(const int test_n : {1,2,3,4,5,10,15,16,30,31,32,33,34,100,200,300}) {
      const Array<const int> test_data = random_permutation(test_n, seed + test_n*4);
      RankedTree<int> l;
      GEODE_ASSERT(l.empty()); // New tree should be empty
      l.test_global_invarients();
      for(const int i : range(test_n)) {
        l.emplace_back(test_data[i]);
        l.test_global_invarients();
        int expected = 0;
        for(const int j : l) {
          GEODE_ASSERT(j == test_data[expected++]);
        }
        GEODE_ASSERT(expected == i+1);
      }

      // Check that we can find values by index
      for(const int i : range(test_n)) {
        const auto iter = l.find_index(i);
        GEODE_ASSERT(iter != l.end() && *iter == test_data[i]);
      }

      // Now go back and remove things one at a time
      for(const auto rev_i : range(1,test_n+1)) {
        const int i = test_data[test_n - rev_i];
        GEODE_ASSERT(!l.empty());
        const auto iter = l.find_last();
        GEODE_ASSERT(*iter == i);
        GEODE_ASSERT(l.erase(iter) == l.end());
        l.test_global_invarients();
        int expected = 0;
        for(const int j : l) {
          GEODE_ASSERT(j == test_data[expected++]);
        }
        GEODE_ASSERT(expected == test_data.size() - rev_i);
      }
      GEODE_ASSERT(l.empty());
    }
  }
}

static void try_insertion_sort() {
  constexpr int test_length = 50;
  const auto test_data = random_permutation(test_length, 0);
  RankedTree<int> l;
  const auto is_sorted = [](RankedTree<int>& l) {
    int prev = -1;
    for(const int i : l) {
      if(!(prev < i)) return false;
      prev = i;
    }
    return true;
  };

  for(const int i : range(test_length)) {
    const int n = test_data[i];
    int n_evaluations = 0;
    const auto p = [&](int e) { n_evaluations += 1; return e < n; };
    l.emplace_in_order(p, n);
    GEODE_ASSERT(n_evaluations <= (integer_log(i) + 2));
    GEODE_ASSERT(is_sorted(l));
    l.test_global_invarients();
  }
  int expected = 0;
  for(const int i : l) {
    GEODE_ASSERT(i == expected);
    expected += 1;
  }
  GEODE_ASSERT(expected == test_length);
}

void ranked_tree_test() {
  try_integer_range();
  try_insertion_sort();
}

} // geode namespace

using namespace geode;

void wrap_ranked_tree() {
  GEODE_FUNCTION(ranked_tree_test)
}
