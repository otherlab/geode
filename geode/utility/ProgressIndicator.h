//#####################################################################
// Class ProgressIndicator
//#####################################################################
#pragma once

#include <geode/utility/config.h>
namespace geode {

class ProgressIndicator {
public:
  int total;
  bool brief;
  int done;
  int percent_done;
  bool print;

  GEODE_CORE_EXPORT ProgressIndicator(const int total=1, const bool brief=false);
  GEODE_CORE_EXPORT void initialize(const int total_input);
  GEODE_CORE_EXPORT bool progress(const int by=1);
};

}
