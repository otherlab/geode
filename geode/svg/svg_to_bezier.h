#pragma once

#include <geode/svg/nanosvg/nanosvg.h>
#include <geode/geometry/Bezier.h>
#include <geode/array/Array.h>
#include <geode/python/Ref.h>
#include <vector>
namespace geode {

using std::vector;
using std::string;

GEODE_CORE_EXPORT vector<Ref<Bezier<2>>> svgfile_to_beziers(const string& filename);
GEODE_CORE_EXPORT vector<Ref<Bezier<2>>> svgstring_to_beziers(const string& svgstring);

}
