#include <other/core/svg/svg_to_bezier.h>
#include <other/core/python/wrap.h>
#include <other/core/python/Class.h>
#include <other/core/python/stl.h>

namespace other {

static vector<Ref<Bezier<2> > > svg_paths_to_beziers(const struct SVGPath* plist) {
  struct Path {
    unsigned int elementIndex;
    vector<Ref<Bezier<2> > > shapes;
    Path(unsigned int _elementIndex)
    : elementIndex(_elementIndex)
    { }
  };

  std::vector<Path> paths;

  Path* path = NULL;
  for (const SVGPath* it = plist; it; it = it->next){
    //allocate new path when elementIndex changes
    if(!path || path->elementIndex != it->elementIndex) {
      OTHER_ASSERT(!path || it->elementIndex < path->elementIndex); //check that these are monotonic so we can't miss subpaths that should be together
      paths.push_back(Path(it->elementIndex));
      path = &paths.back();
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
    if(   it->closed
       || (it->hasFill && bez.knots.size()>2)) { // SVG implicitly closes filled shapes.  Obey that here, unless we only have two knots.
      bez.close();
    }
  }

  vector<Ref<Bezier<2> > > result;
  for(auto& p : paths) {
    for(auto& b : p.shapes)
      result.push_back(b);
  }
  return result;
}

vector<Ref<Bezier<2> > > svgfile_to_beziers(const string& filename) {
  struct SVGPath* plist = svgParseFromFile(filename.c_str(), NULL);
  vector<Ref<Bezier<2> > > all_parts = svg_paths_to_beziers(plist);
  svgDelete(plist);
  return all_parts;
}
vector<Ref<Bezier<2> > > svgstring_to_beziers(const string& svgstring) {
  std::vector<char> str_buf(svgstring.c_str(), svgstring.c_str()+svgstring.size()+1);
  struct SVGPath* plist = svgParse(&str_buf[0], NULL);
  vector<Ref<Bezier<2> > > all_parts = svg_paths_to_beziers(plist);
  svgDelete(plist);
  return all_parts;
}

}

using namespace other;

void wrap_svg_to_bezier() {
  OTHER_FUNCTION(svgfile_to_beziers)
  OTHER_FUNCTION(svgstring_to_beziers)
}
