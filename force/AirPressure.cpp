//#####################################################################
// Class AirPressure
//#####################################################################
#include <other/core/force/AirPressure.h>
#include <other/core/math/constants.h>
#include <other/core/mesh/SegmentMesh.h>
#include <other/core/mesh/TriangleSoup.h>
#include <other/core/python/Class.h>
#include <other/core/structure/Hashtable.h>
#include <other/core/vector/SolidMatrix.h>
#include <other/core/vector/SymmetricMatrix.h>
#include <other/core/utility/Log.h>
namespace other {

using Log::cout;
using std::endl;
typedef real T;
typedef Vector<T,3> TV;

OTHER_DEFINE_TYPE(AirPressure)

AirPressure::AirPressure(Ref<TriangleSoup> mesh,Array<const TV> X,bool closed,int side)
  : mesh(mesh)
  , closed(closed)
  , side(side)
  , skip_rotation_terms(false)
  , initial_volume(side*mesh->volume(X))
  , volume(initial_volume)
{
  OTHER_ASSERT(abs(side)==1);
  temperature = 293; // room temperature
  pressure = 101325; // 1 atm
  if (closed)
    amount = pressure*initial_volume/(ideal_gas_constant*temperature);
  else
    amount = 0;
  // Set up local mesh
  Array<const int> nodes = mesh->nodes_touched();
  Hashtable<int,int> hash;
  for (int i=0;i<nodes.size();i++)
    hash.set(nodes[i],i);
  local_mesh.resize(mesh->elements.size(),false);
  for (int t=0;t<mesh->elements.size();t++)
    for(int i=0;i<3;i++)
      local_mesh[t][i] = hash.get(mesh->elements[t][i]);
}

AirPressure::~AirPressure() {}

void AirPressure::add_frequency_squared(RawArray<T> frequency_squared) const {}

int AirPressure::nodes() const {
  return mesh->nodes();
}

void AirPressure::structure(SolidMatrixStructure& structure) const {
  OTHER_ASSERT(structure.size()>=mesh->nodes());
  if (closed)
    structure.add_outer(1,mesh->nodes_touched());
  for (int t=0;t<mesh->elements.size();t++) {
    int i,j,k;mesh->elements[t].get(i,j,k);
    structure.add_entry(i,j);
    structure.add_entry(j,k);
    structure.add_entry(k,i);
  }
}

void AirPressure::update_position(Array<const TV> X_, bool definite) {
  if (definite && !skip_rotation_terms)
    OTHER_NOT_IMPLEMENTED("Refusing to fix definiteness unless skip_rotation_terms is true");
  OTHER_ASSERT(X_.size()>=mesh->nodes());
  X = X_;
  volume = side*mesh->volume(X);
  if (closed)
    pressure = amount*ideal_gas_constant*temperature/volume;
  Array<const int> nodes = mesh->nodes_touched();
  normals.resize(nodes.size(),false,false);
  normals.zero();
  for (int t=0;t<local_mesh.size();t++) {
    int i,j,k;local_mesh[t].get(i,j,k);
    TV n = cross(X[nodes[j]]-X[nodes[i]],X[nodes[k]]-X[nodes[i]]);
    normals[i]+=n;normals[j]+=n;normals[k]+=n;
  }
}

T AirPressure::elastic_energy() const {
  return closed ? -amount*ideal_gas_constant*temperature*log(volume/initial_volume)
                : -pressure*volume;
}

void AirPressure::add_elastic_force(RawArray<TV> F) const {
  OTHER_ASSERT(X.size()>=mesh->nodes());
  OTHER_ASSERT(F.size()==X.size());
  // F = -dE/dx = -dE/dV dV/dx
  const T dEdV = closed ? -amount*ideal_gas_constant*temperature/volume
                        : -pressure;
  const T factor = -side*dEdV/6;
  // Assuming a closed mesh, we have
  //   V = 1/6 sum_t det(a,b,c)
  //     = 1/6 sum_t det(a-o,b-o,c-o)
  //     = 1/6 sum_t (a-o) . cross(b-o,c-o)
  //   dV/da = 1/6 sum_{t on a} cross(b-o,c-o)
  // This sum is independent of o, so we can pick o=a to get
  //   dV/da = 1/6 sum_{t on a} cross(b-a,c-a)
  // which is just the sum of area weighted triangle normals.
  Array<const int> nodes = mesh->nodes_touched();
  for (int i=0;i<nodes.size();i++)
    F[nodes[i]] += factor*normals[i];
}

void AirPressure::add_elastic_differential(RawArray<TV> dF,RawArray<const TV> dX) const {
  OTHER_ASSERT(dX.size()==X.size());
  OTHER_ASSERT(dF.size()==X.size());
  // We have
  //   dfa/db = d/db (-dE/da) = d/db (-dE/dV dV/da)
  //          = -ddE/dV^2 dV/da dV/db^T - dE/dV ddV/dadb
  //   ddV/dadx = 1/6 d/dx sum_{t on a} cross(b,c)
  // The sum is independent of a, so ddV/da^2 = 0.  If a!=x=b, we have
  //   ddV/dadb = -1/6 sum_{t on a,b} s(a,b) c*
  // where s(a,b) = +1 if the triangle is a,b,c, -1 if it is a,c,b.
  // c* is the cross product action matrix c* x = cross(c,x).  Concretely,
  // if the two triangles joining a,b are adb and abc, we have
  //   6 ddV/dadb = (d-c)*
  //   6 ddV/dbda = (c-d)* = (d-c)*^T
  // and the matrix is symmetric as expected.
  T dEdV,ddE_dVdV;
  if (closed) {
    T nRT = amount*ideal_gas_constant*temperature;
    dEdV = -nRT/volume;
    ddE_dVdV = nRT/sqr(volume);
  } else {
    dEdV = -pressure;
    ddE_dVdV = 0;
  }
  if (ddE_dVdV) {
    T dV = 0;
    Array<const int> nodes = mesh->nodes_touched();
    for (int i=0;i<nodes.size();i++)
      dV += dot(normals[i],dX[nodes[i]]);
    T factor1 = -ddE_dVdV*dV/36;
    for (int i=0;i<nodes.size();i++)
      dF[nodes[i]] += factor1*normals[i];}
  const T factor2 = -side*dEdV/6;
  if (factor2 && !skip_rotation_terms)
    for (int t=0;t<mesh->elements.size();t++){
      int i,j,k;mesh->elements[t].get(i,j,k);
      dF[i] += factor2*(cross(X[j],dX[k])+cross(dX[j],X[k]));
      dF[j] += factor2*(cross(X[k],dX[i])+cross(dX[k],X[i]));
      dF[k] += factor2*(cross(X[i],dX[j])+cross(dX[i],X[j]));
    }
}

void AirPressure::add_elastic_gradient(SolidMatrix<TV>& matrix) const {
  OTHER_ASSERT(X.size()==matrix.size());
  T dEdV;
  if (closed) {
    T nRT = amount*ideal_gas_constant*temperature;
    dEdV = -nRT/volume;
    T ddE_dVdV = nRT/sqr(volume);
    matrix.add_outer(-ddE_dVdV/36,normals);
  } else
    dEdV = -pressure;
  const T factor2 = -side*dEdV/6;
  if (factor2 && !skip_rotation_terms)
    for (int t=0;t<mesh->elements.size();t++) {
      int i,j,k;mesh->elements[t].get(i,j,k);
      matrix.add_entry(i,k,cross_product_matrix(factor2*X[j]));
      matrix.add_entry(j,i,cross_product_matrix(factor2*X[k]));
      matrix.add_entry(k,j,cross_product_matrix(factor2*X[i]));
    }
}

void AirPressure::add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,3> > dFdX) const {
  OTHER_ASSERT(X.size()>=mesh->nodes());
  OTHER_ASSERT(dFdX.size()==X.size());
  if (closed) {
    const T ddE_dVdV = amount*ideal_gas_constant*temperature/sqr(volume);
    const T factor = -ddE_dVdV/36;
    if (factor && !skip_rotation_terms) {
      Array<const int> nodes = mesh->nodes_touched();
      for (int i=0;i<nodes.size();i++)
        dFdX[nodes[i]] += scaled_outer_product(factor,normals[i]);
    }
  }
}

T AirPressure::damping_energy(RawArray<const TV> V) const {
  return 0;
}

void AirPressure::add_damping_force(RawArray<TV> F,RawArray<const TV> V) const {}

void AirPressure::add_damping_gradient(SolidMatrix<TV>& matrix) const {
  if (closed)
    matrix.add_outer(0,normals);
}

T AirPressure::strain_rate(RawArray<const TV> V) const {
  return 0;
}

}
using namespace other;

void wrap_air_pressure() {
  typedef AirPressure Self;
  Class<Self>("AirPressure")
    .OTHER_INIT(Ref<TriangleSoup>,Array<const TV>,bool,int)
    .OTHER_FIELD(temperature)
    .OTHER_FIELD(amount)
    .OTHER_FIELD(pressure)
    .OTHER_FIELD(skip_rotation_terms)
    .OTHER_FIELD(initial_volume)
    ;
}
