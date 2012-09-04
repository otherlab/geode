//#####################################################################
// Class SimplexTree
//#####################################################################
#include <other/core/geometry/SimplexTree.h>
#include <other/core/geometry/Ray.h>
#include <other/core/geometry/Sphere.h>
#include <other/core/geometry/Segment2d.h>
#include <other/core/geometry/Segment3d.h>
#include <other/core/geometry/Triangle2d.h>
#include <other/core/geometry/Triangle3d.h>
#include <other/core/array/IndirectArray.h>
#include <other/core/python/Class.h>
namespace other {
using std::cout;
using std::endl;

typedef real T;

template<> OTHER_DEFINE_TYPE(SimplexTree<Vector<T,2>,1>)
template<> OTHER_DEFINE_TYPE(SimplexTree<Vector<T,2>,2>)
template<> OTHER_DEFINE_TYPE(SimplexTree<Vector<T,3>,1>)
template<> OTHER_DEFINE_TYPE(SimplexTree<Vector<T,3>,2>)

template<class Mesh,class TV> static Array<Box<TV> > boxes(const Mesh& mesh, Array<const TV> X) {
  OTHER_ASSERT(mesh.nodes()<=X.size());
  Array<Box<TV> > boxes(mesh.elements.size(),false);
  for(int t=0;t<mesh.elements.size();t++)
    boxes[t] = bounding_box(X.subset(mesh.elements[t]));
  return boxes;
}

template<class TV,int d> SimplexTree<TV,d>::
SimplexTree(const Mesh& mesh, Array<const TV> X, int leaf_size)
  : Base(RawArray<const Box<TV>>(other::boxes(mesh,X)),leaf_size), mesh(ref(mesh)), X(X), simplices(mesh.elements.size(),false) {
  for (int t=0;t<mesh.elements.size();t++)
    simplices[t] = Simplex(X.subset(mesh.elements[t]));
}

template<class TV,int d> SimplexTree<TV,d>::
~SimplexTree()
{}

template<class TV,int d> void SimplexTree<TV,d>::update() {
  RawArray<const Vector<int,d+1> > elements = mesh->elements;
  for (int t=0;t<elements.size();t++)
    simplices[t] = Simplex(X.subset(elements[t]));
  for (int n : leaves)
    boxes[n] = other::bounding_box(X.subset(prims(n)));
  update_nonleaf_boxes();
}

template<class TV,int d> static void intersection_helper(const SimplexTree<TV,d>& self, Ray<TV>& ray, T thickness_over_two, int node) {
  if (self.boxes[node].lazy_intersects(ray,thickness_over_two)) {
    if (!self.is_leaf(node)) {
      intersection_helper(self,ray,thickness_over_two,2*node+1);
      intersection_helper(self,ray,thickness_over_two,2*node+2);
    } else
      for (int t : self.prims(node))
        if (self.simplices[t].intersection(ray,thickness_over_two))
          ray.aggregate_id = t;
  }
}

template<> void intersection_helper(const SimplexTree<Vector<T,2>,2>& self, Ray<Vector<T,2>>& ray, T thickness_over_two, int node) { OTHER_NOT_IMPLEMENTED(); }
template<> void intersection_helper(const SimplexTree<Vector<T,3>,1>& self, Ray<Vector<T,3>>& ray, T thickness_over_two, int node) { OTHER_NOT_IMPLEMENTED(); }
template<class TV,int d> bool SimplexTree<TV,d>::
intersection(Ray<TV>& ray, T thickness_over_two) const {
  int aggregate_save = ray.aggregate_id;
  ray.aggregate_id = -1;
  intersection_helper(*this,ray,thickness_over_two,0);
  if (ray.aggregate_id<0) {
    ray.aggregate_id = aggregate_save;
    return false;
  }
  return true;
}

template<class TV,int d> void SimplexTree<TV,d>::
intersection(const Sphere<TV>& sphere,Array<int>& hits) const {
  hits.clear();
  intersection_helper(*this,sphere,hits,0);
}

template<class TV,int d> static void intersection_helper(const SimplexTree<TV,d>& self, const Sphere<TV>& sphere, Array<int>& hits, int node) {
  if (self.boxes[node].phi(sphere.center)<=sphere.radius) {
    if (!self.is_leaf(node)) {
      intersection_helper(self,sphere,hits,2*node+1);
      intersection_helper(self,sphere,hits,2*node+2);
    } else
      for (int t : self.prims(node))
        if (self.simplices[t].distance(sphere.center)<=sphere.radius)
          hits.append(t);
  }
}

template<class TV,int d> static void multi_intersection_helper(const SimplexTree<TV,d>& self,Ray<TV>& ray,T thickness_over_two,int node, std::vector<Ray<TV> >& results) {
  if(self.boxes[node].lazy_intersects(ray,thickness_over_two)){
    if(!self.is_leaf(node)){
      multi_intersection_helper(self,ray,thickness_over_two,2*node+1,results);
      multi_intersection_helper(self,ray,thickness_over_two,2*node+2,results);}
    else{
      for(int t : self.prims(node)){
        Ray<TV> copy = ray;
        if(self.simplices[t].intersection(copy,thickness_over_two)){
          copy.aggregate_id=t;
          results.push_back(copy);
        }
      }
    }
  }
}

template<> void multi_intersection_helper(const SimplexTree<Vector<T,2>,2>& self, Ray<Vector<T,2>>& ray, T thickness_over_two, int node, std::vector<Ray<Vector<T,2> > >& results) { OTHER_NOT_IMPLEMENTED(); }
template<> void multi_intersection_helper(const SimplexTree<Vector<T,3>,1>& self, Ray<Vector<T,3>>& ray, T thickness_over_two, int node, std::vector<Ray<Vector<T,3> > >& results) { OTHER_NOT_IMPLEMENTED(); }


template<class TV,int d> std::vector<Ray<TV> > SimplexTree<TV,d>::
intersections(Ray<TV>& ray,T thickness_over_two) const {
  std::vector<Ray<TV> > results;
  multi_intersection_helper(*this,ray,thickness_over_two,0,results);
  return results;
}


// Random directions courtesy of numpy.random.randn.
template<int d> static RawArray<const Vector<T,d>> directions();

template<> RawArray<const Vector<T,2>> directions<2>() {
  typedef Vector<T,2> TV;
  static const TV directions[9] = {
    TV(-0.60969154652467961,0.79263876898392027),TV(0.10835187685622646,-0.99411260467903384),TV(0.74676055230096283,-0.66509298412113849),
    TV(-0.13012888817952251,-0.991497086461257), TV(0.49437504317564707,0.86924870818717759), TV(0.96329975179785676,0.26842799441598403),
    TV(0.51887332761872418,0.85485113902121712), TV(0.42940774149230604,-0.90311073050123625),TV(0.40586459576776618,0.9139332196075749)};
  return RawArray<const TV>(9,directions);
}

template<> RawArray<const Vector<T,3>> directions<3>() {
  typedef Vector<T,3> TV;
  static const TV directions[9] = {
    TV(0.050077866028046092,-0.59798845200904516,0.79993875927967328), TV(-0.094882972991235312,-0.68113055844631043,-0.72598786752049915),TV(-0.21146569967846812,-0.93095746827124071,-0.29765827744159512),
    TV(0.52758229010650814,-0.79574366853717948,0.29740366700658522),  TV(0.38396767739444804,-0.074477799837916317,0.92033791622839078),  TV(-0.55278527979396841,0.43195599711995131,-0.71263065538553194),
    TV(-0.53492079675520676,-0.8446603735997813,-0.020213719822255519),TV(0.26358553445420602,0.49512333260405655,-0.82787411575525371),   TV(0.33993579119653583,-0.85738445772491467,0.38643958067897127)};
  return RawArray<const TV>(9,directions);
}

static inline bool going_out(const Segment<Vector<T,2>>& s, const Vector<T,2>& d) { return cross(d,s.vector())>=0; }
static inline bool going_out(const Triangle<Vector<T,3>>& s, const Vector<T,3>& d) { return dot(s.n,d)>=0; }
static inline bool going_out(const Segment<Vector<T,3>>& s, const Vector<T,3>& d) { OTHER_NOT_IMPLEMENTED(); }
static inline bool going_out(const Triangle<Vector<T,2>>& s, const Vector<T,2>& d) { OTHER_NOT_IMPLEMENTED(); }

template<class Simplex,class TV> static inline bool inside_plane(const Simplex& s, const TV& p) { return going_out(s,s.x0-p); }
static inline bool inside_plane(const Triangle<Vector<T,2>>& s, const Vector<T,2>& p) { OTHER_NOT_IMPLEMENTED(); }

template<class TV,int d> bool SimplexTree<TV,d>::
inside(TV point) const {
  const T small = sqrt(numeric_limits<T>::epsilon());
  // Fire rays in random directions until we hit either nothing or a pure simplex.
  const T epsilon = small*bounding_box().sizes().max();
  for (const TV& dir : directions<TV::m>()) {
    Ray<TV> ray(point,dir,true);
    if (!intersection(ray,epsilon))
      return false; // No intersections, so we must be outside
    const Simplex& simplex = simplices[ray.aggregate_id];
    Vector<T,d+1> w = simplex.barycentric_coordinates(ray.point(ray.t_max));
    if (w.min() > small)
      return going_out(simplex,dir);
  }
  throw ArithmeticError("SimplexTree::inside: all rays were singular");
}

template<class TV,int d> bool SimplexTree<TV,d>::
inside_given_closest_point(TV point, int simplex, Vector<T,d+1> weights) const {
  OTHER_ASSERT(mesh->elements.valid(simplex));
  const T small = sqrt(numeric_limits<T>::epsilon());
  // If the closest point is on a triangle face, we're in luck
  if (weights.min() > small)
    return inside_plane(simplices[simplex],point);
  // Otherwise, fall back to basic inside routine
  return inside(point);
}

template<class TV,int d> static void closest_point_helper(const SimplexTree<TV,d>& self, TV point, int& triangle, T& sqr_distance, int node) {
  if (!self.is_leaf(node)) {
    Vector<T,2> bounds(self.boxes[2*node+1].sqr_distance_bound(point),
                       self.boxes[2*node+2].sqr_distance_bound(point));
    int c = bounds.argmin();
    if (bounds[c]<sqr_distance)
      closest_point_helper(self,point,triangle,sqr_distance,2*node+1+c);
    if (bounds[1-c]<sqr_distance)
      closest_point_helper(self,point,triangle,sqr_distance,2*node+2-c);
  } else
    for (int t : self.prims(node)) {
      T sqr_d = sqr_magnitude(point-self.simplices[t].closest_point(point));
      if (sqr_distance>sqr_d) {
        sqr_distance = sqr_d;
        triangle = t;
      }
    }
}

template<class TV,int d> TV SimplexTree<TV,d>::
closest_point(TV point, int& simplex, Vector<T,d+1>& weights, T max_distance) const {
  OTHER_ASSERT(nodes());
  simplex = -1;
  T sqr_distance = sqr(max_distance);
  closest_point_helper(*this,point,simplex,sqr_distance,0);
  return simplices[simplex].closest_point(point,weights);
}

template<class TV,int d> TV SimplexTree<TV,d>::
closest_point(TV point, T max_distance) const {
  int simplex;
  Vector<T,d+1> weights;
  return closest_point(point, simplex, weights, max_distance);
}

template class SimplexTree<Vector<T,2>,1>;
template class SimplexTree<Vector<T,2>,2>;
template class SimplexTree<Vector<T,3>,1>;
template class SimplexTree<Vector<T,3>,2>;

}
using namespace other;

template<class TV,int d> static void wrap_helper() {
  typedef SimplexTree<TV,d> Self;
  static const string name = format("%sTree%dd",(d==1?"Segment":"Triangle"),TV::m);
  Class<Self>(name.c_str())
    .OTHER_INIT(const typename Self::Mesh&,Array<const TV>,int)
    .OTHER_FIELD(mesh)
    .OTHER_FIELD(X)
    .OTHER_METHOD(update)
    ;
}

void wrap_simplex_tree() {
  wrap_helper<Vector<T,2>,1>();
  wrap_helper<Vector<T,2>,2>();
  wrap_helper<Vector<T,3>,1>();
  wrap_helper<Vector<T,3>,2>();
}
