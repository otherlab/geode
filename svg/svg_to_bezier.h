#pragma once

#include <other/core/svg/nanosvg/nanosvg.h>
#include <other/core/geometry/Bezier.h>
#include <other/core/array/Array.h>
#include <other/core/python/Ref.h>
#include <vector>
namespace other {

using std::vector;
using std::string;

OTHER_CORE_EXPORT vector<Ref<Bezier<2>>> svgfile_to_beziers(const string& filename);
OTHER_CORE_EXPORT vector<Ref<Bezier<2>>> svgstring_to_beziers(const string& svgstring);

}
