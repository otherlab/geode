//#####################################################################
// Class ProgressIndicator
//#####################################################################
#pragma once

#include <other/core/utility/config.h>
namespace other{

class ProgressIndicator {
public:
  int total;
  bool brief;
  int done;
  int percent_done;
  bool print;

  ProgressIndicator(const int total=1, const bool brief=false) OTHER_CORE_EXPORT;
  void initialize(const int total_input) OTHER_CORE_EXPORT;
  bool progress(const int by=1) OTHER_CORE_EXPORT;
};

}
