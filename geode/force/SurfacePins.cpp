//#####################################################################
// Class SurfacePins
//#####################################################################
#include <geode/force/SurfacePins.h>
#include <geode/array/NdArray.h>
#include <geode/array/IndirectArray.h>
#include <geode/structure/Pair.h>
#include <geode/geometry/ParticleTree.h>
#include <geode/geometry/SimplexTree.h>
#include <geode/python/Class.h>
#include <geode/utility/Log.h>
#include <geode/vector/SolidMatrix.h>
#include <geode/vector/SymmetricMatrix.h>
namespace geode {

using Log::cout;
using std::endl;

typedef real T;
typedef Vector<T,3> TV;
GEODE_DEFINE_TYPE(SurfacePins)

SurfacePins::SurfacePins(Array<const int> particles, Array<const T> mass, TriangleMesh& target_mesh, Array<const TV> target_X, NdArray<const T> stiffness, NdArray<const T> damping_ratio)
  : particles(particles)
  , target_mesh(ref(target_mesh))
  , target_X(target_X)
  , mass(mass)
  , k(particles.size(),false)
  , kd(particles.size(),false)
  , node_X(particles.size(),false)
  , target_tree(new_<SimplexTree<TV,2>>(ref(target_mesh),target_X,10))
  , info(particles.size(),false)
{
  max_node = particles.size()?particles.max()+1:0;
  GEODE_ASSERT(mass.size()>=max_node);
  GEODE_ASSERT(target_mesh.nodes()<=target_X.size());
  GEODE_ASSERT(stiffness.rank()==0 || (stiffness.rank()==1 && stiffness.shape[0]==particles.size()));
  GEODE_ASSERT(damping_ratio.rank()==0 || (damping_ratio.rank()==1 && damping_ratio.shape[0]==particles.size()));

  for (int i=0;i<particles.size();i++) {
    int p = particles[i]; 
    T stiffness_ = stiffness.rank()?stiffness[i]:stiffness();
    T damping_ratio_ = damping_ratio.rank()?damping_ratio[i]:damping_ratio();
    k[i] = stiffness_*mass[p];
    kd[i] = 2*damping_ratio_*mass[p]*sqrt(stiffness_);
  }
}

SurfacePins::~SurfacePins() {}

int SurfacePins::nodes() const {
  return max_node;
}

void SurfacePins::structure(SolidMatrixStructure& structure) const {
  GEODE_ASSERT(structure.size()==mass.size());
  GEODE_NOT_IMPLEMENTED("changing matrix topology");
}

void SurfacePins::update_position(Array<const TV> X_, bool definite) {
  GEODE_ASSERT(X_.size()==mass.size());
  node_X.copy(X_.subset(particles));
  if (!node_tree)
    node_tree = new_<ParticleTree<TV>>(node_X,10);
  else
    node_tree->update(); // update for changes to node_X
  // Compute distances and directions
  evaluate_surface_levelset(*node_tree,*target_tree,info,1e10,false);
}

Array<TV> SurfacePins::closest_points(Array<const TV> X) {
  update_position(X,false); 
  Array<TV> closest(particles.size(),false);
  for (int i=0;i<particles.size();i++)
    closest[i] = X[particles[i]]-info[i].phi*info[i].normal;
  return closest;
}

void SurfacePins::add_frequency_squared(RawArray<T> frequency_squared) const {
  GEODE_ASSERT(frequency_squared.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    frequency_squared[p]+=k[i]/mass[p];
  }
}

T SurfacePins::elastic_energy() const {
  T energy = 0;
  for (int i=0;i<particles.size();i++)
    energy += k[i]*sqr(info[i].phi);
  return energy/2;
}

void SurfacePins::add_elastic_force(RawArray<TV> F) const {
  GEODE_ASSERT(F.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    F[p] -= k[i]*info[i].phi*info[i].normal;
  }
}

void SurfacePins::add_elastic_gradient(SolidMatrix<TV>& matrix) const {
  GEODE_ASSERT(matrix.size()==mass.size());
  GEODE_NOT_IMPLEMENTED();
}

void SurfacePins::add_elastic_differential(RawArray<TV> dF, RawArray<const TV> dX) const {
  GEODE_ASSERT(dF.size()==mass.size());
  GEODE_ASSERT(dX.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    dF[p] -= k[i]*dot(dX[p],info[i].normal)*info[i].normal; // Ignores a curvature term if the closest point is on an edge or vertex
  }
}

void SurfacePins::add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,m>> dFdX) const {
  GEODE_ASSERT(dFdX.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    dFdX[p] -= scaled_outer_product(k[i],info[i].normal); // Ignores a curvature term if the closest point is on an edge or vertex
  }
}

T SurfacePins::damping_energy(RawArray<const TV> V) const {
  GEODE_ASSERT(V.size()==mass.size());
  T energy = 0;
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    energy += kd[i]*sqr(dot(V[p],info[i].normal));
  }
  return energy/2;
}

void SurfacePins::add_damping_force(RawArray<TV> F, RawArray<const TV> V) const {
  GEODE_ASSERT(V.size()==mass.size());
  GEODE_ASSERT(F.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    F[p] -= kd[i]*dot(V[p],info[i].normal)*info[i].normal;
  }
}

void SurfacePins::add_damping_gradient(SolidMatrix<TV>& matrix) const {
  GEODE_ASSERT(matrix.size()==mass.size());
  GEODE_NOT_IMPLEMENTED();
}

T SurfacePins::strain_rate(RawArray<const TV> V) const {
  return 0;
}

}
using namespace geode;

void wrap_surface_pins() {
  typedef SurfacePins Self;
  Class<Self>("SurfacePins")
    .GEODE_INIT(Array<const int>,Array<const T>,TriangleMesh&,Array<const TV>,NdArray<const T>,NdArray<const T>)
    .GEODE_METHOD(closest_points)
    ;
}
