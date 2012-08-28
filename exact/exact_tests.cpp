// Unit tests for exact geometric predicates

#include <other/core/exact/predicates.h>
#include <other/core/exact/scope.h>
#include <other/core/array/sort.h>
#include <other/core/math/choice.h>
#include <other/core/python/module.h>
#include <other/core/random/Random.h>
#include <other/core/utility/Log.h>
#include <tr1/unordered_map>
#include <vector>
namespace other {
namespace {

using Log::cout;
using std::endl;
using std::pair;
using std::vector;
using std::ostream;
using std::tr1::unordered_map;

// Slow implementation of infinitesimal expansion arithmetic.
// We use 2 bits for variable exponents, so they can't exceed 3.
struct Expansion {
  typedef uint64_t Monomial;
  unordered_map<Monomial,int64_t> x;

  Expansion() {}

  Expansion(int i, int8_t v) {
    x[0] = v;
    // Later variables get higher degree bits, since they're smaller and therefore should sort last
    x[Monomial(1)<<2*i] = 1;
  }

  Expansion operator+(const Expansion& y) const {
    Expansion r = *this;
    for (auto& t : y.x)
      if (!(r.x[t.first]+=t.second))
        r.x.erase(t.first);
    return r;
  }

  Expansion operator-(const Expansion& y) const {
    Expansion r = *this;
    for (auto& t : y.x)
      if (!(r.x[t.first]-=t.second))
        r.x.erase(t.first);
    return r;
  }

  static Monomial add_degrees(Monomial a, Monomial b) {
    const Monomial odd  = 0x3333333333333333,
                   even = 0xcccccccccccccccc;
    return (((a&odd)+(b&odd))&odd)+(((a&even)+(b&even))&even);
  }

  Expansion operator*(const Expansion& y) const {
    Expansion r;
    for (auto& s : x)
      for (auto& t : y.x)
        r.x[add_degrees(s.first,t.first)] += s.second*t.second;
    // Prune r
    auto it = r.x.begin();
    while (it != r.x.end()) {
      auto prev = it++;
      if (!prev->second)
        r.x.erase(prev);
    }
    return r;
  }

  bool positive() const {
    OTHER_ASSERT(x.size());
    auto it = x.begin();
    Monomial first = it->first;
    bool positive = it->second>0;
    while (++it != x.end())
      if (it->first<first) {
        OTHER_ASSERT(it->second);
        first = it->first;
        positive = it->second>0;
      }
    return positive;
  }

  friend ostream& operator<<(ostream& output, const Expansion& x) {
    vector<pair<Monomial,int64_t>> s(x.x.begin(),x.x.end());
    sort(s);
    bool first = true;
    for (const auto& p : s) {
      if (p.second < 0)
        output << '-';
      else if (!first)
        output << '+';
      first = false;
      bool leading = true;
      if (abs(p.second)!=1) {
        output << abs(p.second);
        leading = false;
      }
      for (int i=0;i<32;i++) {
        const int b = p.first>>2*i&3;
        if (b) {
          if (!leading)
            output << '*';
          leading = false;
          output << 'e'<<i;
          if (b>1)
            output << '^'<<b;
        }
      }
    }
    return output;
  }
};

// Compile time range of integers
template<int n,class... R> struct IRange : public IRange<n-1,R...,mpl::int_<sizeof...(R)>> {};
template<class... R> struct IRange<0,R...> : public Types<R...> {};

template<int d,class Slow,class Fast,class... Once,class... Twice> void test_predicate_helper(const Fast fast, const Slow slow, const int steps, Types<Once...>, Types<Twice...>) {
  const int n = sizeof...(Once);
  const auto random = new_<Random>(7381390412879521731);
  for (int step=0;step<steps;step++) {
    // Generate a random permutation
    Vector<int8_t,n> permutation;
    for (int i=0;i<n;i++)
      permutation[i] = i;
    random->shuffle(permutation);

    // Generate a random set of coefficients, chosen from a randomly sized set of values to create various degeneracies
    int8_t values[n*d];
    int8_t count = min(random->uniform<int8_t>(1,3*n*d/2),int8_t(n*d));
    for (int i=0;i<count;i++)
      values[i] = (int8_t)random->bits<uint8_t>();
    Vector<Vector<float,d>,n> Xf;
    Vector<Vector<Expansion,d>,n> Xe;
    for (int i=0;i<n;i++)
      for (int j=0;j<d;j++) {
        int8_t v = values[random->uniform<int8_t>(0,count)];
        Xf[i][j] = v;
        Xe[i][j] = Expansion(permutation[i]*d+j,v);
      }

    // Compare fast and slow versions of the predicate
    const bool fast_positive = fast(choice<Twice::value&1>(permutation[Twice::value>>1],Xf[Twice::value>>1])...);
    const auto pred = slow(Xe[Once::value]...);
    const bool slow_positive = pred.positive();
    if (fast_positive!=slow_positive) {
      cout << "permutation = "<<Vector<int,n>(permutation)<<endl;
      cout << "Xf = "<<Xf<<endl;
      cout << "Xe = "<<Xe<<endl;
      cout << "fast positive = "<<fast_positive<<endl;
      cout << "slow positive = "<<slow_positive<<endl;
      cout << "predicate = "<<pred<<endl;
      OTHER_ASSERT(false);
    }
  }
}

template<int d,int n,class Slow,class Fast> void test_predicate(const Fast fast, const Slow slow, const int steps) {
  test_predicate_helper<d>(fast,slow,steps,IRange<n>(),IRange<2*n>());
}

typedef Expansion E;
typedef Vector<E,2> EV2;

E slow_triangle_oriented(const EV2& p0, const EV2& p1, const EV2& p2) {
  return cross(p1-p0,p2-p0);
}

E slow_segment_directions_oriented(const EV2& a0, const EV2& a1, const EV2& b0, const EV2& b1) {
  return cross(a1-a0,b1-b0);
}

E slow_segment_intersections_ordered_helper(const EV2& a0, const EV2& a1, const EV2& b0, const EV2& b1, const EV2& c0, const EV2& c1) {
  const auto da = a1-a0;
  const auto db = b1-b0;
  const auto dc = c1-c0;
  return cross(c0-a0,dc)*cross(da,dc)-cross(b0-a0,db)*cross(da,db);
}

void predicate_tests(const int steps) {
  IntervalScope scope;
  {
    Log::Scope scope("triangle_oriented");
    test_predicate<2,3>(triangle_oriented,slow_triangle_oriented,steps);
  } {
    Log::Scope scope("segment_directions_oriented");
    test_predicate<2,4>(segment_directions_oriented,slow_segment_directions_oriented,steps);
  } {
    Log::Scope scope("segment_intersections_ordered_helper");
    test_predicate<2,6>(segment_intersections_ordered_helper,slow_segment_intersections_ordered_helper,steps);
  }
}

}
}
using namespace other;

void wrap_exact_tests() {
  OTHER_FUNCTION(predicate_tests)
}
