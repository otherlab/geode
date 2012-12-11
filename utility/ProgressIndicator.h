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

  OTHER_CORE_EXPORT ProgressIndicator(const int total=1, const bool brief=false) ;
  OTHER_CORE_EXPORT void initialize(const int total_input) ;
  OTHER_CORE_EXPORT bool progress(const int by=1) ;
};

}
