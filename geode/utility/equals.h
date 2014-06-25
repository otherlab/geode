#pragma once

#include <geode/utility/tr1.h>
#include <vector>
#include <set>
namespace geode {

template<class T> struct Equals {
  inline static bool eval(T const &a, T const &b) {
    return a==b;
  }
};

template<class A, class B, class C> struct Equals<unordered_set<A,B,C>> {
  inline static bool eval(unordered_set<A,B,C> const &s1, unordered_set<A,B,C> const &s2) {
    if (s1.size() != s2.size())
      return false;

    for (typename unordered_set<A,B,C>::const_iterator it = s1.begin(); it != s1.end(); ++it) {
      if (!s2.count(*it)) {
        return false;
      }
    }
    return true;
  }
};

template<class A, class B, class C> struct Equals<unordered_map<A,B,C>> {
  inline static bool eval(unordered_map<A,B,C> const &s1, unordered_map<A,B,C> const &s2) {
    if (s1.size() != s2.size())
      return false;

    for (typename unordered_map<A,B,C>::const_iterator it = s1.begin(); it != s1.end(); ++it) {
      if (!s2.count(it->first)) {
        return false;
      }else{
        if(s2.find(it->first)->second != it->second){
          return false;
        }
      }

    }
    return true;
  }
};

template<class A, class B> struct Equals<std::pair<A,B>> {
  inline static bool eval(std::pair<A,B> const &s1, std::pair<A,B> const &s2) {
    return Equals<A>::eval(s1.first,s2.first) && Equals<B>::eval(s1.second,s2.second);
  }
};

template<class A> struct Equals<std::vector<A>> {
  inline static bool eval(std::vector<A> const &s1, std::vector<A> const &s2) {
    if (s1.size() != s2.size())
      return false;
    for (size_t i = 0; i < s1.size(); ++i)
      if (!Equals<A>::eval(s1[i], s2[i]))
        return false;
    return true;
  }
};

}

