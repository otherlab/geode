#include <geode/force/LinearFiniteVolumeHex.h>
#include <geode/array/view.h>
#include <geode/math/Factorial.h>
#include <geode/utility/const_cast.h>
#include <geode/vector/normalize.h>
#include <geode/vector/SolidMatrix.h>
#include <geode/vector/SymmetricMatrix.h>
#include <geode/vector/UpperTriangularMatrix.h>
namespace geode {

typedef real T;
typedef Vector<T,3> TV;

LinearFiniteVolumeHex::LinearFiniteVolumeHex(const StrainMeasureHex& strain, const T density, const T youngs_modulus, const T poissons_ratio, const T rayleigh_coefficient)
  : strain(ref(strain))
  , youngs_modulus(youngs_modulus)
  , poissons_ratio(poissons_ratio)
  , rayleigh_coefficient(rayleigh_coefficient)
  , density(density) {
  X = strain.rest_X;
}

LinearFiniteVolumeHex::~LinearFiniteVolumeHex() {}

void LinearFiniteVolumeHex::update_position(Array<const TV> X_, bool definite) {
  GEODE_ASSERT(X_.size()>=strain->nodes);
  X = X_;
  mu_lambda();
}

Vector<T,2> LinearFiniteVolumeHex::mu_lambda() const {
  GEODE_ASSERT(poissons_ratio!=-1 && poissons_ratio!=.5);
  T lambda = youngs_modulus*poissons_ratio/((1+poissons_ratio)*(1-2*poissons_ratio));
  T mu = youngs_modulus/(2*(1+poissons_ratio));
  return vec(mu,lambda);
}

T LinearFiniteVolumeHex::elastic_energy() const {
  GEODE_ASSERT(X.size()>=strain->nodes);
  T energy = 0;
  T mu,lambda;mu_lambda().get(mu,lambda);
  T half_lambda = (T).5*lambda;
  for (int h=0;h<strain->elements.size();h++)
    for (int g=0;g<8;g++) {
      SymmetricMatrix<T,m> S = symmetric_part(strain->gradient(X,h,g))-1;
      energy += strain->DmH_det[h][g]*(mu*S.sqr_frobenius_norm()+half_lambda*sqr(S.trace()));
    }
  return energy;
}

void LinearFiniteVolumeHex::add_elastic_force(RawArray<TV> F) const {
  GEODE_ASSERT(X.size()>=strain->nodes && F.size()==X.size());
  T mu,lambda;mu_lambda().get(mu,lambda);
  T two_mu = 2*mu;
  T two_mu_plus_m_lambda = 2*mu+m*lambda;
  for (int h=0;h<strain->elements.size();h++)
    for (int g=0;g<8;g++) {
      SymmetricMatrix<T,m> S_plus_one = symmetric_part(strain->gradient(X,h,g));
      T scale = strain->DmH_det[h][g];
      SymmetricMatrix<T,m> scaled_stress = scale*two_mu*S_plus_one+scale*(lambda*S_plus_one.trace()-two_mu_plus_m_lambda);
      strain->distribute_stress(F,scaled_stress,h,g);
    }
}

void LinearFiniteVolumeHex::add_differential_helper(RawArray<TV> dF, RawArray<const TV> dX, T scale) const {
  GEODE_ASSERT(X.size()>=strain->nodes && dF.size()==X.size() && dX.size()==X.size());
  T mu,lambda;(scale*mu_lambda()).get(mu,lambda);
  T two_mu = 2*mu;
  for (int h=0;h<strain->elements.size();h++)
    for (int g=0;g<8;g++) {
      SymmetricMatrix<T,m> d_strain = symmetric_part(strain->gradient(dX,h,g));
      T scale = strain->DmH_det[h][g];
      SymmetricMatrix<T,m> d_scaled_stress = scale*two_mu*d_strain+scale*lambda*d_strain.trace();
      strain->distribute_stress(dF,d_scaled_stress,h,g);
    }
}

void LinearFiniteVolumeHex::add_elastic_differential(RawArray<TV> dF, RawArray<const TV> dX) const {
  add_differential_helper(dF,dX,1);
}

void LinearFiniteVolumeHex::add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,m>> dFdX) const {
  GEODE_NOT_IMPLEMENTED();
}

T LinearFiniteVolumeHex::damping_energy(RawArray<const TV> V) const {
  GEODE_ASSERT(X.size()>=strain->nodes);
  T energy = 0;
  T beta,alpha;(rayleigh_coefficient*mu_lambda()).get(beta,alpha);
  T half_alpha = (T).5*alpha;
  for (int h=0;h<strain->elements.size();h++)
    for (int g=0;g<8;g++) {
      SymmetricMatrix<T,m> S = symmetric_part(strain->gradient(V,h,g));
      energy += strain->DmH_det[h][g]*(beta*S.sqr_frobenius_norm()+half_alpha*sqr(S.trace()));
    }
  return energy;
}

void LinearFiniteVolumeHex::add_damping_force(RawArray<TV> F, RawArray<const TV> V) const {
  add_differential_helper(F,V,rayleigh_coefficient);
}

void LinearFiniteVolumeHex::add_frequency_squared(RawArray<T> frequency_squared) const {
  T mu,lambda;mu_lambda().get(mu,lambda);
  T stiffness = lambda+2*mu;
  Hashtable<int,T> particle_frequency_squared;
  for (int h=0;h<strain->elements.size();h++) {
    T elastic_squared = stiffness/(sqr(strain->DmH_minimum_altitude(h))*density);
    for (int k=0;k<8;k++) {
      T& data = particle_frequency_squared.get_or_insert(strain->elements[h][k]);
      data = max(data,elastic_squared);
    }
  }
  for (auto& it : particle_frequency_squared)
    frequency_squared[it.x] += it.y;
}

T LinearFiniteVolumeHex::strain_rate(RawArray<const TV> V) const {
  GEODE_ASSERT(V.size()>=strain->nodes);
  T strain_rate = 0;
  for (int h=0;h<strain->elements.size();h++)
    for (int g=0;g<8;g++)
      strain_rate = max(strain_rate,strain->gradient(V,h,g).maxabs());
  return strain_rate;
}

int LinearFiniteVolumeHex::nodes() const {
  return strain->nodes;
}

void LinearFiniteVolumeHex::structure(SolidMatrixStructure& structure) const {
  GEODE_NOT_IMPLEMENTED();
}

void LinearFiniteVolumeHex::add_elastic_gradient(SolidMatrix<TV>& matrix) const {
  GEODE_NOT_IMPLEMENTED();
}

void LinearFiniteVolumeHex::add_damping_gradient(SolidMatrix<TV>& matrix) const {
  GEODE_NOT_IMPLEMENTED();
}

}
