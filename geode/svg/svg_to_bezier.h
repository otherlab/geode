#pragma once

#include <geode/svg/nanosvg/nanosvg.h>
#include <geode/geometry/Bezier.h>
#include <geode/array/Array.h>
#include <geode/utility/Ref.h>
#include <vector>
namespace geode {

using std::vector;
using std::string;

// a path (all subpaths) and its style
class SVGStyledPath: public Object {
public:
  GEODE_NEW_FRIEND

  unsigned int fillColor;
  unsigned int strokeColor;
  float strokeWidth;
  int hasFill;
  int fillRule; // 1 for nonzero (default), 2 for evenodd
  bool hasStroke;
  string CSSclass;

  unsigned int elementIndex;
  vector<Ref<Bezier<2>>> shapes;

protected:
  SVGStyledPath(unsigned int _elementIndex)
  : elementIndex(_elementIndex)
  {}
};

GEODE_CORE_EXPORT vector<Ref<SVGStyledPath>> svgfile_to_styled_beziers(const string& filename);
GEODE_CORE_EXPORT vector<Ref<SVGStyledPath>> svgstring_to_styled_beziers(const string& svgstring);

GEODE_CORE_EXPORT vector<Ref<Bezier<2>>> svgfile_to_beziers(const string& filename);
GEODE_CORE_EXPORT vector<Ref<Bezier<2>>> svgstring_to_beziers(const string& svgstring);

}
