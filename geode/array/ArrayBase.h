//#####################################################################
// Class ArrayBase
//#####################################################################
#pragma once

#include <geode/array/forward.h>
#include <geode/array/ArrayAbs.h>
#include <geode/array/ArrayDifference.h>
#include <geode/array/ArrayLeftMultiple.h>
#include <geode/array/ArrayNegation.h>
#include <geode/array/ArrayPlusScalar.h>
#include <geode/array/ArrayProduct.h>
#include <geode/array/ArraySum.h>
#include <geode/structure/forward.h>
#include <geode/math/hash.h>
#include <geode/math/max.h>
#include <geode/math/maxabs.h>
#include <geode/math/maxmag.h>
#include <geode/math/min.h>
#include <geode/math/minmag.h>
#include <geode/math/inverse.h>
#include <geode/math/isnan.h>
#include <geode/vector/magnitude.h>
#include <geode/vector/Dot.h>
#include <geode/vector/ScalarPolicy.h>
#include <geode/utility/STATIC_ASSERT_SAME.h>
#include <geode/utility/type_traits.h>
#include <boost/mpl/identity.hpp>
#include <iostream>
#include <limits>
namespace geode {

using std::type_info;
template<class T,class TArray> class ArrayBase;

template<class TArray,class Enabler=void> struct CanonicalizeConstArray{typedef TArray type;};

template<class TArray1,class TArray2> struct SameArrayCanonical { static bool same_array(const TArray1& array1,const TArray2& array2) {
  static_assert(!is_same<TArray1,TArray2>::value,"");
  return false;
}};

template<class TArray> struct SameArrayCanonical<TArray,TArray> { static bool same_array(const TArray& array1,const TArray& array2) {
  return TArray::same_array(array1,array2);
}};

template<class TA1,class TA2> struct SameArray : public SameArrayCanonical<typename CanonicalizeConstArray<TA1>::type,typename CanonicalizeConstArray<TA2>::type>{};

// Throw IndexError with nice formatting.  Defined in Array.cpp
GEODE_CORE_EXPORT void GEODE_NORETURN(out_of_bounds(const type_info& type, const int size, const int i)) GEODE_COLD;

template<class T_,class TArray>
class ArrayBase {
  struct Unusable{};
  typedef typename remove_const<T_>::type T;
public:
  typedef T Element;

  // for stl
  typedef T value_type;
  typedef T_* iterator;
  typedef T_* const_iterator;
  typedef int difference_type;
  typedef int size_type;

  typedef typename ScalarPolicy<T>::type Scalar;

protected:
  ArrayBase() GEODE_ALWAYS_INLINE {}
  ArrayBase(const ArrayBase&) {}
  ~ArrayBase() {}
public:

  TArray& derived() {
    return static_cast<TArray&>(*this);
  }

  const TArray& derived() const {
    return static_cast<const TArray&>(*this);
  }

  template<class TArray1> static bool same_array(const TArray1& array1, const TArray1& array2) {
    return &array1==&array2;
  }

  template<class TArray1,class TArray2> static bool same_array(const TArray1& array1,const TArray2& array2) {
    return SameArray<TArray1,TArray2>::same_array(array1,array2);
  }

protected:
  TArray& operator=(const ArrayBase& source) {
    TArray& self = derived();
    int m = self.size();
    const TArray& source_ = source.derived();
    assert(m==source_.size());
    if (!TArray::same_array(self,source_))
      for (int i=0;i<m;i++)
        self[i] = source_[i];
    return self;
  }

  template<class TArray1> TArray& operator=(const TArray1& source) {
    STATIC_ASSERT_SAME(T,typename TArray1::Element);
    TArray& self = derived();
    int m = self.size();
    assert(m==source.size());
    if (!TArray::same_array(self,source))
      for (int i=0;i<m;i++)
        self[i] = source[i];
    return self;
  }
public:

  template<class TIndices> IndirectArray<const TArray,TIndices&> subset(const TIndices& indices) const {
    return IndirectArray<const TArray,TIndices&>(derived(),indices);
  }

  IndirectArray<TArray,ARange> prefix(const int prefix_size) const;

private:
  typedef typename mpl::if_<is_class<T>,T,Unusable>::type TIfClass;
public:

  template<class TField,TField TIfClass::* field> ProjectedArray<TArray,FieldProjector<TIfClass,TField,field> > project() const {
    return ProjectedArray<TArray,FieldProjector<Element,TField,field> >(derived());
  }

  ProjectedArray<const TArray,IndexProjector> project(const int index) const;

private:
  template<class S> struct ElementOf{typedef typename S::Element type;};
  typedef typename mpl::if_<IsVector<T>,ElementOf<T>,mpl::identity<Unusable> >::type::type ElementOfT;
  typedef typename mpl::if_<is_const<T_>,const ElementOfT,ElementOfT>::type TOfT;
public:

  bool empty() const {
    return !derived().size();
  }

  template<class TArray1> bool operator==(const TArray1& v) const {
    STATIC_ASSERT_SAME(T,typename TArray1::Element);
    const TArray& self = derived();
    int m = self.size();
    if (m!=v.size()) return false;
    for (int i=0;i<m;i++) if(self[i]!=v[i]) return false;
    return true;
  }

  template<class TArray1> bool operator!=(const TArray1& v) const {
    return !(*this==v);
  }

  template<class TArray1> const TArray& operator+=(const ArrayBase<T,TArray1>& v) const {
    const TArray& self = derived();
    int m = self.size();
    const TArray1& v_ = v.derived();
    assert(m==v_.size());
    for (int i=0;i<m;i++) self[i] += v_[i];
    return self;
  }

  const TArray& operator+=(const T& a) const { // This could be merged with the version below if windows wasn't broken
    const TArray& self = derived();
    int m = self.size();
    for (int i=0;i<m;i++) self[i] += a;
    return self;
  }

  template<class T2> typename enable_if<IsScalar<T2>,const TArray&>::type operator+=(const T2& a) const {
    const TArray& self = derived();
    int m = self.size();
    for (int i=0;i<m;i++) self[i] += a;
    return self;
  }

  template<class TArray1> const TArray& operator-=(const ArrayBase<T,TArray1>& v) const {
    const TArray& self = derived();
    int m = self.size();
    const TArray1& v_ = v.derived();
    assert(m==v_.size());
    for (int i=0;i<m;i++) self[i] -= v_[i];
    return self;
  }

  template<class T2> typename enable_if<mpl::or_<IsScalar<T2>,is_same<T,T2> >,const TArray&>::type operator-=(const T2& a) const {
    const TArray& self = derived();
    int m = self.size();
    for (int i=0;i<m;i++) self[i] -= a;
    return self;
  }

  template<class T2,class TArrayT2> const TArray& operator*=(const ArrayBase<T2,TArrayT2>& v) const {
    const TArray& self = derived();
    int m = self.size();
    const TArrayT2& v_ = v.derived();
    assert(m==v_.size());
    for (int i=0;i<m;i++) self[i] *= v_[i];
    return self;
  }

  template<class T2> typename enable_if<mpl::or_<IsScalar<T2>,is_same<T,T2> >,const TArray&>::type operator*=(const T2& a) const {
    const TArray& self = derived();int m = self.size();
    for (int i=0;i<m;i++) self[i] *= a;
    return self;
  }

  template<class T2,class TArrayT2> const TArray& operator/=(const ArrayBase<T2,TArrayT2>& a) const {
    const TArray& self = derived();
    int m = self.size();
    const TArrayT2& a_ = a.derived();
    assert(m==a_.size());
    for (int i=0;i<m;i++) {
      assert(a_(i));
      self[i] /= a_[i];
    }
    return self;
  }

  template<class T2> typename enable_if<mpl::or_<IsScalar<T2>,is_same<T,T2> >,const TArray&>::type operator/=(const T2& a) const {
    return *this *= inverse(a);
  }

  void negate() const {
    const TArray& self = derived();
    int m = self.size();
    for (int i=0;i<m;i++) self[i] = -self[i];
  }

  T_& front() const {
    return derived()[0];
  }

  T_& back() const {
    const TArray& self = derived();
    return self[self.size()-1];
  }

  int find(const T& element) const {
    const TArray& self = derived();
    int m = self.size();
    for (int i=0;i<m;i++) if (self[i]==element) return i;
    return -1;
  }

  bool contains(const T& element) const {
    const TArray& self = derived();
    int m = self.size();
    for (int i=0;i<m;i++) if (self[i]==element) return true;
    return false;
  }

  bool contains_only(const T& element) const {
    const TArray& self = derived();
    int m = self.size();
    for (int i=0;i<m;i++) if (self[i]!=element) return false;
    return true;
  }

  int count_matches(const T& value) const {
    const TArray& self = derived();
    int m = self.size();
    int count = 0;
    for (int i=0;i<m;i++) if (self[i]==value) count++;
    return count;
  }

  int count_true() const {
    const TArray& self = derived();
    int m = self.size();
    int count = 0;
    for (int i=0;i<m;i++) if (self[i]) count++;
    return count;
  }

  int count_false() const {
    const TArray& self = derived();
    int m = self.size();
    int count = 0;
    for (int i=0;i<m;i++) if (!self[i]) count++;
    return count;
  }

  void fill(const T& constant) const {
    const TArray& self = derived();
    int m = self.size();
    for (int i=0;i<m;i++) self[i] = constant;
  }

  Array<Element> copy() const {
    Array<Element> copy;
    copy.copy(derived());
    return copy;
  }

  template<class T2,class TArray1> void copy(const T2 constant,const TArray1& array) {
    copy(constant*array);
  }

  template<class T2,class TArray1,class TArray2> void copy(const T2 c1,const TArray1& v1,const TArray2& v2) {
    copy(c1*v1+v2);
  }

  template<class T2,class TArray1,class TArray2> void copy(const T2 c1,const TArray1& v1,const T2 c2,const TArray2& v2) {
    copy(c1*v1+c2*v2);
  }

  template<class T2,class TArray1,class TArray2,class TArray3> void copy(const T2 c1,const TArray1& v1,const T2 c2,const TArray2& v2,const T2 c3,const TArray3& v3) {
    copy(c1*v1+c2*v2+c3*v3);
  }

  void copy_or_fill(const T& constant) { // for occasional templatization purposes
    fill(constant);
  }

  template<class TArray1> void copy_or_fill(const TArray1& source) { // for occasional templatization purposes
    copy(source);
  }

  T max() const {
    const TArray& self = derived();
    T result = self[0];
    int m = self.size();
    for (int i=1;i<m;i++) result = geode::max(result,self[i]);
    return result;
  }

  T maxabs() const {
    const TArray& self = derived();
    T result = T();
    int m = self.size();
    for (int i=0;i<m;i++) result = geode::max(result,abs(self[i]));
    return result;
  }

  T maxmag() const {
    const TArray& self = derived();
    T result = T();
    int m = self.size();
    for (int i=1;i<m;i++) result = geode::maxmag(result,self[i]);
    return result;
  }

  int argmax() const {
    const TArray& self = derived();
    int result = 0,
        m = self.size();
    for (int i=1;i<m;i++) if (self[i]>self[result]) result = i;
    return result;
  }

  T min() const {
    const TArray& self = derived();
    T result = self[0];
    int m = self.size();
    for (int i=1;i<m;i++) result = geode::min(result,self[i]);
    return result;
  }

  T minmag() const {
    const TArray& self = derived();
    T result = self[0];
    int m = self.size();
    for (int i=1;i<m;i++) result = geode::minmag(result,self[i]);
    return result;
  }

  int argmin() const {
    const TArray& self = derived();
    int result = 0,
        m = self.size();
    for (int i=1;i<m;i++) if (self[i]<self[result]) result = i;
    return result;
  }

  T sum() const {
    const TArray& self = derived();
    T result = T();
    int m = self.size();
    for (int i=0;i<m;i++) result += self[i];
    return result;
  }

  T product() const {
    const TArray& self = derived();
    T result = 1;
    int m = self.size();
    for (int i=0;i<m;i++) result *= self[i];
    return result;
  }

  T mean() const {
    const TArray& self = derived();
    int m = self.size();
    return m?sum()/Scalar(m):T();
  }

  Scalar sqr_magnitude() const {
    const TArray& self = derived();
    Scalar result = 0;
    int m = self.size();
    for (int i=0;i<m;i++) result += geode::sqr_magnitude(self[i]);
    return result;
  }

  Scalar magnitude() const {
    return sqrt(sqr_magnitude());
  }

private:
  Scalar min_magnitude_helper(mpl::true_ is_scalar) const {
    const TArray& self = derived();
    T result = std::numeric_limits<T>::infinity();
    int m = self.size();
    for (int i=0;i<m;i++) result=geode::min(result,abs(self[i]));
    return result;
  }

  Scalar max_magnitude_helper(mpl::true_ is_scalar) const {
    const TArray& self = derived();
    T result = 0;
    int m = self.size();
    for (int i=0;i<m;i++) result=geode::max(result,abs(self[i]));
    return result;
  }

  Scalar min_magnitude_helper(mpl::false_ is_scalar) const {
    return sqrt(min_sqr_magnitude());
  }

  Scalar max_magnitude_helper(mpl::false_ is_scalar) const {
    return sqrt(max_sqr_magnitude());
  }

  Scalar min_sqr_magnitude_helper(mpl::true_ is_scalar) const {
    return sqr(min_magnitude());
  }

  Scalar max_sqr_magnitude_helper(mpl::true_ is_scalar) const {
    return sqr(max_magnitude());
  }

  Scalar min_sqr_magnitude_helper(mpl::false_ is_scalar) const {
    const TArray& self = derived();
    Scalar result = std::numeric_limits<Scalar>::infinity();
    int m = self.size();
    for (int i=0;i<m;i++) result = geode::min(result,self[i].sqr_magnitude());
    return result;
  }

  Scalar max_sqr_magnitude_helper(mpl::false_ is_scalar) const {
    const TArray& self = derived();
    Scalar result = 0;
    int m = self.size();
    for (int i=0;i<m;i++) result = geode::max(result,self[i].sqr_magnitude());
    return result;
  }

  int argmax_magnitude_helper(mpl::true_ is_scalar) const {
    const TArray& self = derived();
    int m = self.size();
    Scalar max = -1;
    int argmax = -1;
    for (int i=0;i<m;i++) {
      Scalar current = abs(self[i]);
      if (max<current) {
        max = current;
        argmax = i;
      }
    }
    return argmax;
  }

  int argmax_magnitude_helper(mpl::false_ is_scalar) const {
    const TArray& self = derived();
    int m = self.size();
    Scalar max = -1;
    int argmax = -1;
    for (int i=0;i<m;i++) {
      Scalar current = self[i].sqr_magnitude();
      if (max<current) {
        max = current;
        argmax = i;
      }
    }
    return argmax;
  }
public:

  Scalar min_magnitude() const {
    return min_magnitude_helper(IsScalar<T>());
  }

  Scalar max_magnitude() const {
    return max_magnitude_helper(IsScalar<T>());
  }

  Scalar min_sqr_magnitude() const {
    return min_sqr_magnitude_helper(IsScalar<T>());
  }

  Scalar max_sqr_magnitude() const {
    return max_sqr_magnitude_helper(IsScalar<T>());
  }

  int argmax_magnitude() const {
    return argmax_magnitude_helper(IsScalar<T>());
  }

  template<class TArray1,class TArrayInt> static void permute(const TArray1& source,TArray1& destination,const TArrayInt& permutation) {
    STATIC_ASSERT_SAME(T,typename TArray1::Element);
    STATIC_ASSERT_SAME(int,typename TArrayInt::Element);
    int m = permutation.size();
    for (int i=0;i<m;i++) destination[i] = source[permutation[i]];
  }

  template<class TArray1,class TArrayInt> static void unpermute(const TArray1& source,TArray1& destination,const TArrayInt& permutation) {
    STATIC_ASSERT_SAME(T,typename TArray1::Element);
    STATIC_ASSERT_SAME(int,typename TArrayInt::Element);
    int m = permutation.size();
    for (int i=0;i<m;i++) destination[permutation[i]] = source[i];
  }

  template<class TArray1> void remove_sorted_indices(const TArray1& index) {
    STATIC_ASSERT_SAME(int,typename TArray1::Element);
    TArray& self = derived();
    int m = self.size(),
        index_m = index.size();
    if (!index_m) return;
    for (int kk=0;kk<index_m-1;kk++) {
      assert(unsigned(index(kk))<unsigned(index(kk+1)));
      for (int i=index(kk)-kk;i<index(kk+1)-kk-1;i++)
        self[i] = self[i+kk];
    }
    assert(unsigned(index(index_m-1))<unsigned(m));
    for (int i=index(index_m)-index_m;i<m-index_m;i++)
      self[i] = self[i+index_m];
    self.resize(m-index_m);
  }

  template<class TArray1> void remove_sorted_indices_lazy(const TArray1& index) {
    STATIC_ASSERT_SAME(int,typename TArray1::Element);
    TArray& self = derived();
    int index_m = index.size();
    if (!index_m) return;
    for (int k=index_m-1;k>=0;k--)
      self.remove_index_lazy(index(k));
    self.compact();
  }

  void reverse() {
    TArray& self = derived();
    int m = self.size();
    for (int i=0;i<m/2;i++) swap(self[i],self[m-1-i]);
  }

  T_* begin() const { // for stl
    return derived().data();
  }

  T_* end() const { // for stl
    return derived().data()+derived().size();
  }

  T& at(const int i) const {
    const TArray& self = derived();
    const int m = self.size();
    if (unsigned(i) >= unsigned(m))
      out_of_bounds(typeid(T),m,i);
    return self[i];
  }
};

template<class T,class TArray> inline bool isnan(const ArrayBase<T,TArray>& a_) {
  const TArray& a = a_.derived();
  int m = a.size();
  for (int i=0;i<m;i++)
    if (isnan(a[i])) return true;
  return false;
}

template<class T> static inline bool isnan(const RawArray<T>& a) {
  return isnan((const ArrayBase<T,RawArray<T>>&)a);
}

template<class T,int d> static inline bool isnan(const RawArray<T,d>& a) {
  return isnan(a.flat);
}

template<class T,int d> static inline bool isnan(const Array<T,d>& a) {
  return isnan(a.raw());
}

template<class T,class TArray1,class TArray2> inline typename ScalarPolicy<T>::type dot(const ArrayBase<T,TArray1>& a1_, const ArrayBase<T,TArray2>& a2_) {
  typedef typename ScalarPolicy<typename remove_const<T>::type>::type Scalar;
  const TArray1& a1 = a1_.derived();
  const TArray2& a2 = a2_.derived();
  assert(a1.size()==a2.size());
  Scalar result = 0;
  int m = a1.size();
  for (int i=0;i<m;i++) result += geode::dot(a1[i],a2[i]);
  return result;
}

struct InnerUnusable{};

template<class T,class TM,class TArray1,class TArray2> inline typename ScalarPolicy<T>::type
inner_product(const ArrayBase<TM,TArray1>& m_, const ArrayBase<T,TArray2>& a1_, const ArrayBase<T,TArray2>& a2_, typename enable_if<IsScalar<TM>,InnerUnusable>::type unusable=InnerUnusable()) {
  typedef typename ScalarPolicy<T>::type Scalar;
  const TArray1& m = m_.derived();
  const TArray2 &a1 = a1_.derived(),
                &a2 = a2_.derived();
  assert(a1.size()==a2.size());
  Scalar result=0;
  int size = a1.size();
  for (int i=0;i<size;i++) result += m[i]*geode::dot(a1[i],a2[i]);
  return result;
}

template<class T,class TM,class TArray1,class TArray2> inline typename ScalarPolicy<T>::type
inner_product(const ArrayBase<TM,TArray1>& m_,const ArrayBase<T,TArray2>& a1_,const ArrayBase<T,TArray2>& a2_,typename disable_if<IsScalar<TM>,InnerUnusable>::type unusable=InnerUnusable()) {
  typedef typename ScalarPolicy<T>::type Scalar;
  const TArray1& m = m_.derived();
  const TArray2 &a1 = a1_.derived(),
                &a2 = a2_.derived();
  Scalar result = 0;
  int size = a1.size();
  assert(m.size()==size && a2.size()==size);
  for (int i=0;i<size;i++) result += m(i).inner_product(a1[i],a2[i]);
  return result;
}

template<class T,class TArray> inline std::ostream& operator<<(std::ostream& output, const ArrayBase<T,TArray>& a) {
  const TArray& a_ = a.derived();
  const int m = a_.size();
  output << '[';
  for (int i=0;i<m;i++) {
    if (i)
      output << ',';
    output << a_[i];
  }
  return output<<']';
}

template<class T,class TArray> static inline Hash hash_reduce(const ArrayBase<T,TArray>& key) {
  return hash_array(key.derived());
}

template<class T,class TA> const TA& concatenate(const ArrayBase<T,TA>& a) {
  return a.derived();
}

template<class T0,class T1,class TA0,class TA1> Array<typename remove_const<T0>::type> concatenate(const ArrayBase<T0,TA0>& a0_, const ArrayBase<T1,TA1>& a1_) {
  STATIC_ASSERT_SAME(typename remove_const<T0>::type, typename remove_const<T1>::type);
  const auto& a0 = a0_.derived();
  const auto& a1 = a1_.derived();
  const int m0 = a0.size(),
            m1 = a1.size();
  Array<typename remove_const<T0>::type> result(m0+m1,false);
  result.slice(0,m0) = a0;
  result.slice(m0,m0+m1) = a1;
  return result;
}

}
#include <geode/array/arange.h>
#include <geode/array/ProjectedArray.h>
#include <geode/array/Array.h>
namespace geode {

template<class T_,class TArray> inline IndirectArray<TArray,ARange> ArrayBase<T_,TArray>::prefix(const int prefix_size) const {
  assert(prefix_size<=derived().size());
  return IndirectArray<const TArray,ARange>(derived(),ARange(prefix_size));
}

template<class T_,class TArray> inline ProjectedArray<const TArray,IndexProjector> ArrayBase<T_,TArray>::project(const int index) const {
  return ProjectedArray<TArray,IndexProjector>(derived(),IndexProjector(index));
}

}
