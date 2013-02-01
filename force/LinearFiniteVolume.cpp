#include <other/core/force/LinearFiniteVolume.h>
#include <other/core/array/view.h>
#include <other/core/math/Factorial.h>
#include <other/core/python/Class.h>
#include <other/core/utility/const_cast.h>
#include <other/core/vector/normalize.h>
#include <other/core/vector/SolidMatrix.h>
#include <other/core/vector/SymmetricMatrix.h>
#include <other/core/vector/UpperTriangularMatrix.h>
namespace other{

typedef real T;
template<> OTHER_DEFINE_TYPE(LinearFiniteVolume<Vector<T,2>,2>)
template<> OTHER_DEFINE_TYPE(LinearFiniteVolume<Vector<T,3>,2>)
template<> OTHER_DEFINE_TYPE(LinearFiniteVolume<Vector<T,3>,3>)

namespace{
template<class T,int d> inline Vector<T,d> normal(const Matrix<T,d>& A) {
  return Vector<T,d>();
}
template<class T> inline Vector<T,3> normal(const Matrix<T,3,2>& A) {
  return normalized(A.weighted_normal());
}
}
template<class TV,int d> LinearFiniteVolume<TV,d>::
LinearFiniteVolume(Array<const Vector<int,d+1> > elements,Array<const TV> X_,const T density,const T youngs_modulus,const T poissons_ratio,const T rayleigh_coefficient)
    :elements(elements),youngs_modulus(youngs_modulus),poissons_ratio(poissons_ratio),rayleigh_coefficient(rayleigh_coefficient),nodes_(elements.size()?1+scalar_view(elements).max():0),density(density)
{
  update_position(X_,false);
  const_cast_(Dm_inverse) = Array<Matrix<T,d,m> >(elements.size(),false);
  Bm_scales.resize(elements.size(),false,false);
  if((int)m>(int)d) normals.resize(elements.size(),false,false);
  for(int t=0;t<elements.size();t++){
    Matrix<T,m,d> Dm = Ds(X,t);
    T scale = Dm.parallelepiped_measure();
    Matrix<T,d,m> Dm_ct = Dm.cofactor_matrix().transposed();
    if((int)m>(int)d) normals[t] = normal(Dm);
    OTHER_ASSERT(scale>0);
    const_cast_(Dm_inverse[t]) = Dm_ct/scale;
    Bm_scales[t] = -(T)1/Factorial<d>::value*scale;}
}

template<class TV,int d> LinearFiniteVolume<TV,d>::
~LinearFiniteVolume()
{}

template<class TV,int d> int LinearFiniteVolume<TV,d>::nodes() const {
  return nodes_;
}

template<class TV,int d> void LinearFiniteVolume<TV,d>::
update_position(Array<const TV> X_,bool definite) {
  OTHER_ASSERT(X_.size()>=nodes_);
  X = X_;
  mu_lambda();
}

template<class TV,int d> Vector<typename TV::Scalar,2> LinearFiniteVolume<TV,d>::
mu_lambda() const {
  OTHER_ASSERT(poissons_ratio!=-1 && poissons_ratio!=.5);
  T lambda = youngs_modulus*poissons_ratio/((1+poissons_ratio)*(1-2*poissons_ratio));
  T mu = youngs_modulus/(2*(1+poissons_ratio));
  return vec(mu,lambda);
}

template<class TV,int d> typename TV::Scalar LinearFiniteVolume<TV,d>::
elastic_energy() const {
  OTHER_ASSERT(X.size()>=nodes_);
  T energy = 0;
  T mu,lambda;mu_lambda().get(mu,lambda);
  T half_lambda = (T).5*lambda;
  for(int t=0;t<elements.size();t++){
    SymmetricMatrix<T,m> strain = symmetric_part(Ds(X,t)*Dm_inverse[t])-1; 
    if((int)m>(int)d) strain += outer_product(normals[t]);
    energy -= Bm_scales[t]*(mu*strain.sqr_frobenius_norm()+half_lambda*sqr(strain.trace()));}
  return energy;
}

template<class TV,int d> void LinearFiniteVolume<TV,d>::
add_elastic_force(RawArray<TV> F) const {
  OTHER_ASSERT(X.size()>=nodes_ && F.size()==X.size());
  T mu,lambda;mu_lambda().get(mu,lambda);
  T two_mu = 2*mu;
  T two_mu_plus_m_lambda = 2*mu+m*lambda;
  for(int t=0;t<elements.size();t++){
    SymmetricMatrix<T,m> strain_plus_one = symmetric_part(Ds(X,t)*Dm_inverse[t]);
    if((int)m>(int)d) strain_plus_one += outer_product(normals[t]);
    SymmetricMatrix<T,m> scaled_stress = Bm_scales[t]*two_mu*strain_plus_one+Bm_scales[t]*(lambda*strain_plus_one.trace()-two_mu_plus_m_lambda);
    StrainMeasure<TV,d>::distribute_force(F,elements[t],scaled_stress.times_transpose(Dm_inverse[t]));}
}

template<class TV,int d> void LinearFiniteVolume<TV,d>::
add_differential_helper(RawArray<TV> dF,RawArray<const TV> dX,T scale) const {
  OTHER_ASSERT(X.size()>=nodes_ && dF.size()==X.size() && dX.size()==X.size());
  T mu,lambda;(scale*mu_lambda()).get(mu,lambda);
  T two_mu = 2*mu;
  for(int t=0;t<elements.size();t++){
    SymmetricMatrix<T,m> d_strain = symmetric_part(Ds(dX,t)*Dm_inverse[t]);
    SymmetricMatrix<T,m> d_scaled_stress = Bm_scales[t]*two_mu*d_strain+Bm_scales[t]*lambda*d_strain.trace();
    StrainMeasure<TV,d>::distribute_force(dF,elements[t],d_scaled_stress.times_transpose(Dm_inverse[t]));}
}

template<class TV,int d> void LinearFiniteVolume<TV,d>::
add_elastic_differential(RawArray<TV> dF,RawArray<const TV> dX) const {
  add_differential_helper(dF,dX,1);
}

template<class TV,int d> void LinearFiniteVolume<TV,d>::
add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,m> > dFdX) const {
  OTHER_NOT_IMPLEMENTED();
}

template<class TV,int d> typename TV::Scalar LinearFiniteVolume<TV,d>::
damping_energy(RawArray<const TV> V) const {
  OTHER_ASSERT(X.size()>=nodes_);
  T energy = 0;
  T beta,alpha;(rayleigh_coefficient*mu_lambda()).get(beta,alpha);
  T half_alpha = (T).5*alpha;
  for(int t=0;t<elements.size();t++){
    SymmetricMatrix<T,m> strain = symmetric_part(Ds(V,t)*Dm_inverse[t]); 
    energy -= Bm_scales[t]*(beta*strain.sqr_frobenius_norm()+half_alpha*sqr(strain.trace()));}
  return energy;
}

template<class TV,int d> void LinearFiniteVolume<TV,d>::
add_damping_force(RawArray<TV> F,RawArray<const TV> V) const {
  add_differential_helper(F,V,rayleigh_coefficient);
}

namespace{
template<class T,int d> inline T simplex_minimum_altitude(const Matrix<T,d>& Dm_inverse) {
  return Dm_inverse.inverse().simplex_minimum_altitude();
}
template<class T> inline T simplex_minimum_altitude(const Matrix<T,2,3>& Dm_inverse) {
  return Dm_inverse.transposed().R_from_QR_factorization().inverse().simplex_minimum_altitude();
}
}
template<class TV,int d> void LinearFiniteVolume<TV,d>::
add_frequency_squared(RawArray<T> frequency_squared) const {
  T mu,lambda;mu_lambda().get(mu,lambda);
  T stiffness = lambda+2*mu;
  Hashtable<int,T> particle_frequency_squared;
  for(int t=0;t<elements.size();t++){
    T elastic_squared = stiffness/(sqr(simplex_minimum_altitude(Dm_inverse[t]))*density);
    for(int i=0;i<d+1;i++){
      T& data = particle_frequency_squared.get_or_insert(elements[t][i]);
      data = max(data,elastic_squared);}}
  for (auto& it : particle_frequency_squared)
    frequency_squared[it.key] += it.data;
}

template<class TV,int d> typename TV::Scalar LinearFiniteVolume<TV,d>::
strain_rate(RawArray<const TV> V) const {
  OTHER_ASSERT(V.size()>=nodes_);
  T strain_rate=0;
  for(int t=0;t<elements.size();t++)
    strain_rate=max(strain_rate,(Ds(V,t)*Dm_inverse[t]).maxabs());
  return strain_rate;
}

template<class TV,int d> void LinearFiniteVolume<TV,d>::
structure(SolidMatrixStructure& structure) const {
  OTHER_NOT_IMPLEMENTED(); // Remove once add_elastic/damping_gradient are implemented
  for(int t=0;t<elements.size();t++){
    Vector<int,d+1> nodes = elements[t];
    for(int i=0;i<nodes.size();i++) for(int j=i+1;j<nodes.size();j++)
      structure.add_entry(i,j);}
}

template<class TV,int d> void LinearFiniteVolume<TV,d>::
add_elastic_gradient(SolidMatrix<TV>& matrix) const {
  OTHER_NOT_IMPLEMENTED();
}

template<class TV,int d> void LinearFiniteVolume<TV,d>::
add_damping_gradient(SolidMatrix<TV>& matrix) const {
  OTHER_NOT_IMPLEMENTED();
}

template class LinearFiniteVolume<Vector<T,2>,2>;
template class LinearFiniteVolume<Vector<T,3>,2>;
template class LinearFiniteVolume<Vector<T,3>,3>;
}

void wrap_linear_finite_volume() {
  using namespace other;

  {typedef LinearFiniteVolume<Vector<T,2>,2> Self;
  Class<Self>("LinearFiniteVolume2d")
    .OTHER_INIT(Array<const Vector<int,3> >,Array<const Vector<T,2> >,T,T,T,T)
    ;}

  {typedef LinearFiniteVolume<Vector<T,3>,2> Self;
  Class<Self>("LinearFiniteVolumeS3d")
    .OTHER_INIT(Array<const Vector<int,3> >,Array<const Vector<T,3> >,T,T,T,T)
    ;}

  {typedef LinearFiniteVolume<Vector<T,3>,3> Self;
  Class<Self>("LinearFiniteVolume3d")
    .OTHER_INIT(Array<const Vector<int,4> >,Array<const Vector<T,3> >,T,T,T,T)
    ;}
}
