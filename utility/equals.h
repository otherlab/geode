#pragma once

#include <tr1/unordered_set>
#include <vector>
#include <set>

namespace other {

template<class T> struct equals {
  inline static bool eval(T const &a, T const &b) {
    return a==b;
  }
};

template<class A, class B, class C> struct equals<std::tr1::unordered_set<A,B,C>> {
  inline static bool eval(std::tr1::unordered_set<A,B,C> const &s1, std::tr1::unordered_set<A,B,C> const &s2) {
    if (s1.size() != s2.size())
      return false;

    for (auto x : s1) {
      if (!s2.count(x)) {
        return false;
      }
    }
    return true;
  }
};

template<class A, class B> struct equals<std::pair<A,B>> {
  inline static bool eval(std::pair<A,B> const &s1, std::pair<A,B> const &s2) {
    return equals<A>::eval(s1.first,s2.first) && equals<B>::eval(s1.second,s2.second);
  }
};

template<class A> struct equals<std::vector<A>> {
  inline static bool eval(std::vector<A> const &s1, std::vector<A> const &s2) {
    if (s1.size() != s2.size())
      return false;
    for (int i = 0; i < s1.size(); ++i)
      if (!equals<A>::eval(s1[i], s2[i]))
        return false;
    return true;
  }
};

}

