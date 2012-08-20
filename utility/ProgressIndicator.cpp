//#####################################################################
// Class ProgressIndicator
//#####################################################################
#include <other/core/utility/ProgressIndicator.h>
#include <other/core/utility/Log.h>
namespace other {

ProgressIndicator::ProgressIndicator(const int total, const bool brief)
  : total(total), brief(brief), done(0), percent_done(0), print(true) {}

void ProgressIndicator::initialize(const int total_input) {
  total = total_input;
  done = percent_done = 0;
}

bool ProgressIndicator::progress(const int by) {
  done += by;
  int new_percent_done = 100*done/total;
  if (new_percent_done>percent_done){
    percent_done = new_percent_done;
    if (print) {
      if (brief)
        Log::cout<<'.'<<std::flush;
      else
        Log::cout<<percent_done<<"% "<<std::flush;
      if (percent_done==100)
        Log::cout<<std::endl;
    }
    return true;
  }
  return false;
}

}
