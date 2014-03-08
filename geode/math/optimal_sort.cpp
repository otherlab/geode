// Optimal sorting networks up to n = 10

#include <geode/math/optimal_sort.h>
#include <geode/math/popcount.h>
#include <geode/python/wrap.h>
#include <geode/utility/format.h>
namespace geode {

static void optimal_sort_test() {
  // From Bundala and Zavodny (2013). Optimal Sorting Networks. http://arxiv.org/pdf/1310.6271v2.pdf.
  static const int optimal[16+2] = {0,0,1,3,3,5,5,6,6,7,7,8,8,9,9,9,9};
  #define L() levels++;
  #define CL() mask = 0;
  #define C(i,j) { \
    static_assert(i<j,"Incorrect comparator order"); \
    const auto m = (1<<i)|(1<<j); \
    GEODE_ASSERT(!(mask&m),format("n %d, ij %d %d",n_,i,j)); \
    mask |= m; \
    const auto b = a; \
    a &= ~((~b>>j&1)<<i); \
    a |=    (b>>i&1)<<j; \
  }
  #define N(...)
  #define TEST(n) { \
    GEODE_UNUSED const int n_ = n; \
    int levels = 0; \
    GEODE_SORT_NETWORK(n,N,L) \
    GEODE_ASSERT(levels==optimal[n]); \
    for (uint32_t i=0;i<(1<<n);i++) { \
      uint32_t a = i; \
      GEODE_UNUSED uint32_t mask; \
      GEODE_SORT_NETWORK(n,C,CL) \
      const int p = popcount(i); \
      if (a != uint32_t(((1<<p)-1)<<(n-p))) \
        throw AssertionError(format("optimal_sort_test: n %d, i %d, a %d",n,i,a)); \
    } \
  }
  TEST(0)
  TEST(1)
  TEST(2)
  TEST(3)
  TEST(4)
  TEST(5)
  TEST(6)
  TEST(7)
  TEST(8)
  TEST(9)
  TEST(10)
}

}
using namespace geode;

void wrap_optimal_sort() {
  GEODE_FUNCTION(optimal_sort_test)
}
