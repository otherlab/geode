#pragma once

#include <string>
#include <cctype>

namespace geode {

inline string to_lower(string const &S) {
  string s(S);
  for (auto &c : s)
    c = tolower(c);
  return s;
}

inline string to_upper(string const &s) {
  string S(s);
  for (auto &c : S)
    c = toupper(c);
  return S;
}

}
