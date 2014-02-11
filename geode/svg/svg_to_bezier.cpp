#include <geode/svg/svg_to_bezier.h>
#include <geode/python/wrap.h>
#include <geode/python/Class.h>
#include <geode/python/to_python.h>
#include <geode/python/Ref.h>
#include <geode/python/stl.h>
#include <geode/utility/stl.h>

namespace geode {

GEODE_DEFINE_TYPE(SVGStyledPath)

static vector<Ref<SVGStyledPath>> svg_paths_to_beziers(const struct SVGPath* plist) {
  std::vector<Ref<SVGStyledPath>> paths;

  Ptr<SVGStyledPath> path;
  for (const SVGPath* it = plist; it; it = it->next){
    //allocate new path only when elementIndex changes
    if(!path || path->elementIndex != it->elementIndex) {
      GEODE_ASSERT(!path || it->elementIndex < path->elementIndex); //check that these are monotonic so we can't miss subpaths that should be together
      paths.push_back(new_<SVGStyledPath>(it->elementIndex));
      path = paths.back();

      // fill in style info
      path->fillColor = it->fillColor;
      path->strokeColor = it->strokeColor;
      path->strokeWidth = it->strokeWidth;
      path->hasFill = it->hasFill;
      path->fillRule = it->fillRule;
      path->hasStroke = it->hasStroke;
      path->CSSclass = it->CSSclass;
    }

    //make new subpath
    path->shapes.push_back(new_<Bezier<2> >());
    Bezier<2>& bez = *path->shapes.back();
    Vector<real,2> p(it->bezpts[0],it->bezpts[1]);
    Vector<real,2> t(it->bezpts[2],it->bezpts[3]);
    bez.append_knot(p,p,t);
    for (int i = 3; i < it->nbezpts; i+=3){
      Vector<real,2> tan_in(it->bezpts[2*(i-1)], it->bezpts[2*(i-1)+1]);
      Vector<real,2> pt(it->bezpts[2*i], it->bezpts[2*i+1]);
      Vector<real,2> tan_out = (i<it->nbezpts-1) ? Vector<real,2>(it->bezpts[2*(i+1)], it->bezpts[2*(i+1)+1]) : pt;
      bez.append_knot(pt,tan_in,tan_out);
    }
    if(it->closed
       || (it->hasFill && bez.knots.size()>2)) { // SVG implicitly closes filled shapes.  Obey that here, unless we only have two knots
      auto last = bez.knots.end();
      --last;
      auto prev = last;
      --prev;
      (last->second->pt - prev->second->pt).magnitude() > 1e-3*bez.bounding_box().sizes().magnitude() ? bez.close() : bez.fuse_ends();
    }
  }
  return paths;
}

vector<Ref<SVGStyledPath>> svgfile_to_styled_beziers(const string& filename) {
  struct SVGPath* plist = svgParseFromFile(filename.c_str(), NULL);
  auto paths = svg_paths_to_beziers(plist);
  svgDelete(plist);
  return paths;
}

vector<Ref<SVGStyledPath>> svgstring_to_styled_beziers(const string& svgstring) {
  std::vector<char> str_buf(svgstring.c_str(), svgstring.c_str()+svgstring.size()+1);
  struct SVGPath* plist = svgParse(&str_buf[0], NULL);
  auto paths = svg_paths_to_beziers(plist);
  svgDelete(plist);
  return paths;
}

vector<Ref<Bezier<2>>> svgfile_to_beziers(const string& filename) {
  auto styled_parts = svgfile_to_styled_beziers(filename);
  vector<Ref<Bezier<2>>> parts;
  for (auto part : styled_parts) {
    extend(parts, part->shapes);
  }
  return parts;
}

vector<Ref<Bezier<2>>> svgstring_to_beziers(const string& svgstring) {
  auto styled_parts = svgstring_to_styled_beziers(svgstring);
  vector<Ref<Bezier<2>>> parts;
  for (auto part : styled_parts) {
    extend(parts, part->shapes);
  }
  return parts;
}

}

using namespace geode;

void wrap_svg_to_bezier() {
  typedef SVGStyledPath Self;
  Class<Self>("SVGStyledPath")
    .GEODE_FIELD(fillColor)
    .GEODE_FIELD(strokeColor)
    .GEODE_FIELD(hasFill)
    .GEODE_FIELD(hasStroke)
    .GEODE_FIELD(CSSclass)
    .GEODE_FIELD(shapes)
    ;

  GEODE_FUNCTION(svgfile_to_styled_beziers)
  GEODE_FUNCTION(svgstring_to_styled_beziers)
  GEODE_FUNCTION(svgfile_to_beziers)
  GEODE_FUNCTION(svgstring_to_beziers)
}
