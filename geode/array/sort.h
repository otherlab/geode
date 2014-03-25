// sort and stable_sort
#pragma once

#include <geode/array/Array.h>
#include <geode/math/min.h>
#include <algorithm>
#include <functional>
namespace geode {

// Comparison function objects

template<class TArray,class TCompare=std::less<typename TArray::value_type>> struct IndirectCompare {
  const TArray& array;
  TCompare comparison;

  IndirectCompare(const TArray& array)
    : array(array) {}

  IndirectCompare(const TArray& array, const TCompare& comparison)
    : array(array), comparison(comparison) {}

  template<class T> bool operator()(const T& index1, const T& index2) const {
    return comparison(array(index1),array(index2));
  }
};

template<class TArray> static inline IndirectCompare<TArray> indirect_comparison(const TArray& array) {
  return IndirectCompare<TArray>(array);
}

template<class TArray,class TComparison> static inline IndirectCompare<TArray,TComparison> indirect_comparison(const TArray& array,const TComparison& comparison) {
  return IndirectCompare<TArray,TComparison>(array,comparison);
}

template<class T,class TField> struct FieldCompare {
  TField T::*field;

  FieldCompare(TField T::*field_input)
    : field(field_input) {}

  bool operator()(const T& x1,const T& x2) const {
    return x1.*field<x2.*field;
  }
};

template<class T,class TField> static inline FieldCompare<T,TField> field_comparison(TField T::*field) {
  return FieldCompare<T,TField>(field);
}

template<class TA> static inline bool lex_less(const TA& a0, const TA& a1) {
  const int m0 = a0.size(),
            m1 = a1.size(),
            m = min(m0,m1);
  for (int i=0;i<m;i++)
    if (a0[i]!=a1[i])
      return a0[i]<a1[i];
  return m0<m1;
}

struct LexicographicCompare {
  template<class TA> bool operator()(const TA& a0, const TA& a1) const {
    return lex_less(a0,a1);
  }
};

// sort and stable_sort

template<class TArray,class TCompare> static inline void sort(TArray& array, const TCompare& comparison) {
  std::sort(array.begin(),array.end(),comparison);
}

template<class TArray,class TCompare> static inline void sort(const TArray& array, const TCompare& comparison) {   
  std::sort(array.begin(),array.end(),comparison);
}

template<class TArray,class TCompare> static inline void stable_sort(TArray& array,const TCompare& comparison) {
  std::stable_sort(array.begin(),array.end(),comparison);
}

template<class TArray> static inline void sort(TArray& array) {
  sort(array,std::less<typename TArray::value_type>());
}

template<class TArray> static inline void sort(const TArray& array) {
  sort(array,std::less<typename TArray::value_type>());
}

template<class TArray> static inline void stable_sort(TArray& array) {
  stable_sort(array,std::less<typename TArray::value_type>());
}

}
