//#####################################################################
// Class BindingSprings
//#####################################################################
#include <geode/force/BindingSprings.h>
#include <geode/array/NdArray.h>
#include <geode/array/IndirectArray.h>
#include <geode/array/view.h>
#include <geode/structure/Hashtable.h>
#include <geode/vector/SolidMatrix.h>
#include <geode/vector/SymmetricMatrix.h>
namespace geode {

typedef real T;
typedef Vector<T,3> TV;

template<int k> BindingSprings<k>::BindingSprings(Array<const int> nodes, Array<const Vector<int,k>> parents, Array<const Vector<T,k>> weights, Array<const T> mass, NdArray<const T> stiffness, NdArray<const T> damping_ratio)
  : mass(mass)
  , info(nodes.size(),uninit)
{
  GEODE_ASSERT(nodes.size()==parents.size());
  GEODE_ASSERT(nodes.size()==weights.size());
  const int max_node = nodes.size()?max(nodes.max(),scalar_view(parents).max())+1:0;
  GEODE_ASSERT(mass.size()>=max_node);
  GEODE_ASSERT(stiffness.rank()==0 || (stiffness.rank()==1 && stiffness.shape[0]==nodes.size()));
  GEODE_ASSERT(damping_ratio.rank()==0 || (damping_ratio.rank()==1 && damping_ratio.shape[0]==nodes.size()));

  for (int b=0;b<info.size();b++) {
    BindingInfo<k>& I = info[b];
    I.node = nodes[b];
    I.parents = parents[b];
    I.weights = weights[b];
    GEODE_ASSERT(abs(I.weights.sum()-1)<1e-5);
    T inv_binding_mass = (sqr(I.weights)/Vector<T,k>(mass.subset(I.parents))).sum();
    T harmonic_mass = 1/(1/mass[I.node]+inv_binding_mass);
    T stiffness_ = stiffness.rank()?stiffness[b]:stiffness();
    T damping_ratio_ = damping_ratio.rank()?damping_ratio[b]:damping_ratio();
    I.k = stiffness_*harmonic_mass;
    I.kd = 2*damping_ratio_*harmonic_mass*sqrt(stiffness_);
  }
}

template<int k> BindingSprings<k>::~BindingSprings() {}

template<int k> int BindingSprings<k>::nodes() const {
  return mass.size();
}

template<int k> void BindingSprings<k>::structure(SolidMatrixStructure& structure) const {
  GEODE_ASSERT(mass.size()==structure.size());
  for (int b=0;b<info.size();b++) {
    const BindingInfo<k>& I = info[b];
    for (int i=0;i<k;i++) {
      structure.add_entry(I.node,I.parents[i]);
      for (int j=i+1;j<k;j++)
        structure.add_entry(I.parents[i],I.parents[j]);
    }
  }
}

template<int k> void BindingSprings<k>::update_position(Array<const TV> X_,bool definite) {
  GEODE_ASSERT(X_.size()==mass.size());
  X = X_;
}

template<int k> void BindingSprings<k>::add_frequency_squared(RawArray<T> frequency_squared) const {
  // Ignore CFL contribution for now
}

template<int k> static inline TV gather(RawArray<const TV> X, const BindingInfo<k>& I) {
  TV r = X[I.node];
  for (int i=0;i<k;i++)
    r -= I.weights[i]*X[I.parents[i]];
  return r;
}

template<int k> static inline void scatter(RawArray<TV> F,const BindingInfo<k>& I,const TV& f) {
  F[I.node] -= f;
  for (int i=0;i<k;i++)
    F[I.parents[i]] += I.weights[i]*f;
}

template<int k> static inline void add_entries(SolidMatrix<TV>& matrix,const BindingInfo<k>& I,T scale) {
  matrix.add_entry(I.node,-scale);
  for (int i=0;i<k;i++) {
    T sw = scale*I.weights[i];
    matrix.add_entry(I.node,I.parents[i],sw);
    matrix.add_entry(I.parents[i],-sw*I.weights[i]);
    for (int j=i+1;j<k;j++)
      matrix.add_entry(I.parents[i],I.parents[j],-sw*I.weights[j]);
  }
}

template<int k> T BindingSprings<k>::elastic_energy() const {
  T energy = 0;
  for (int b=0;b<info.size();b++) {
    const BindingInfo<k>& I = info[b];
    energy += I.k*sqr_magnitude(gather(X,I));
  }
  return energy/2;
}

template<int k> void BindingSprings<k>::add_elastic_force(RawArray<TV> F) const {
  add_elastic_differential(F,X);
}

template<int k> void BindingSprings<k>::add_elastic_differential(RawArray<TV> dF, RawArray<const TV> dX) const {
  GEODE_ASSERT(dF.size()==mass.size());
  GEODE_ASSERT(dX.size()==mass.size());
  for (int b=0;b<info.size();b++) {
    const BindingInfo<k>& I = info[b];
    scatter(dF,I,I.k*gather(dX,I));
  }
}

template<int k> void BindingSprings<k>::add_elastic_gradient(SolidMatrix<TV>& matrix) const {
  GEODE_ASSERT(matrix.size()==mass.size());
  for (int b=0;b<info.size();b++) {
    const BindingInfo<k>& I = info[b];
    add_entries(matrix,I,I.k);
  }
}

template<int k> void BindingSprings<k>::add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,m>> dFdX) const {
  GEODE_ASSERT(dFdX.size()==mass.size());
  for (int b=0;b<info.size();b++) {
    const BindingInfo<k>& I = info[b];
    dFdX[I.node] -= I.k;
    for (int i=0;i<k;i++)
      dFdX[I.parents[i]] -= sqr(I.weights[i])*I.k;
  }
}

template<int k> T BindingSprings<k>::damping_energy(RawArray<const TV> V) const {
  GEODE_ASSERT(V.size()==mass.size());
  T energy = 0;
  for (int b=0;b<info.size();b++) {
    const BindingInfo<k>& I = info[b];
    energy += I.kd*sqr_magnitude(gather(V,I));
  }
  return energy/2;
}

template<int k> void BindingSprings<k>::add_damping_force(RawArray<TV> F,RawArray<const TV> V) const {
  GEODE_ASSERT(V.size()==mass.size());
  GEODE_ASSERT(F.size()==mass.size());
  for (int b=0;b<info.size();b++) {
    const BindingInfo<k>& I = info[b];
    scatter(F,I,I.kd*gather(V,I));
  }
}

template<int k> void BindingSprings<k>::add_damping_gradient(SolidMatrix<TV>& matrix) const {
  GEODE_ASSERT(matrix.size()==mass.size());
  for (int b=0;b<info.size();b++) {
    const BindingInfo<k>& I = info[b];
    add_entries(matrix,I,I.kd);
  }
}

template<int k> T BindingSprings<k>::strain_rate(RawArray<const TV> V) const {
  return 0;
}

}
