//#####################################################################
// Class MovFile
//#####################################################################
#pragma once

#include <geode/array/Array.h>
namespace geode {

class QtAtom;

class MovWriter : public Object {
public:
  GEODE_NEW_FRIEND
  typedef Object Base;
  typedef real T;
private:
  int frames_per_second;
  int width,height;
  FILE* fp;
  QtAtom* current_mov;
  Array<int> sample_offsets;
  Array<int> sample_lengths;

protected:
  GEODE_CORE_EXPORT MovWriter(const std::string& filename,const int frames_per_second=24);
public:
  GEODE_CORE_EXPORT ~MovWriter();
  GEODE_CORE_EXPORT void add_frame(const Array<Vector<T,3>,2>& image);
  GEODE_CORE_EXPORT void write_footer();
  GEODE_CORE_EXPORT static bool enabled();
};

}
