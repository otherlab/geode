//#####################################################################
// Class MovFile
//#####################################################################
#pragma once

#include <other/core/array/Array.h>
namespace other {

class QtAtom;

class MovWriter : public Object {
public:
  OTHER_DECLARE_TYPE
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
  OTHER_EXPORT MovWriter(const std::string& filename,const int frames_per_second=24);
public:
  OTHER_EXPORT ~MovWriter();
  OTHER_EXPORT void add_frame(const Array<Vector<T,3>,2>& image);
  OTHER_EXPORT void write_footer();
  OTHER_EXPORT static bool enabled();
};

}
