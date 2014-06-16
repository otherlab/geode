//#####################################################################
#include <geode/force/FiniteVolume.h>
#include <geode/force/AnisotropicConstitutiveModel.h>
#include <geode/force/IsotropicConstitutiveModel.h>
#include <geode/force/DiagonalizedStressDerivative.h>
#include <geode/force/DiagonalizedIsotropicStressDerivative.h>
#include <geode/force/PlasticityModel.h>
#include <geode/force/StrainMeasure.h>
#include <geode/structure/Hashtable.h>
#include <geode/math/Factorial.h>
#include <geode/python/Class.h>
#include <geode/vector/DiagonalMatrix.h>
#include <geode/vector/Matrix.h>
#include <geode/vector/SolidMatrix.h>
#include <geode/vector/SymmetricMatrix.h>
#include <geode/vector/UpperTriangularMatrix.h>
#include <geode/utility/Log.h>
namespace geode {

using Log::cout;
using std::endl;

typedef real T;
template<> GEODE_DEFINE_TYPE(FiniteVolume<Vector<T,2>,2>)
template<> GEODE_DEFINE_TYPE(FiniteVolume<Vector<T,3>,2>)
template<> GEODE_DEFINE_TYPE(FiniteVolume<Vector<T,3>,3>)

template<class TV,int d> FiniteVolume<TV,d>::FiniteVolume(StrainMeasure<T,d>& strain, T density, ConstitutiveModel<T,d>& model, Ptr<PlasticityModel<T,d>> plasticity)
  : strain(ref(strain))
  , density(density)
  , model(ref(model))
  , plasticity(plasticity)
  , Be_scales(strain.elements.size(),uninit)
  , stress_derivatives_valid(false)
  , definite(false) {
  for (int t=0;t<Be_scales.size();t++)
    Be_scales[t] = -(T)1/Factorial<d>::value/strain.Dm_inverse[t].determinant();
  isotropic = dynamic_cast<IsotropicConstitutiveModel<T,d>*>(&model);
  anisotropic = dynamic_cast<AnisotropicConstitutiveModel<T,d>*>(&model);
}

template<class TV,int d> FiniteVolume<TV,d>::~FiniteVolume() {}

template<class TV,int d> int FiniteVolume<TV,d>::nodes() const {
  return strain->nodes;
}

template<class TV,int d> void FiniteVolume<TV,d>::structure(SolidMatrixStructure& structure) const {
  for (int t=0;t<strain->elements.size();t++) {
    Vector<int,d+1> nodes = strain->elements[t];
    for (int i=0;i<nodes.size();i++)
      for (int j=i+1;j<nodes.size();j++)
        structure.add_entry(nodes[i],nodes[j]);
  }
}

template<int d,int m> static inline typename enable_if_c<m==d,const Matrix<T,d,d>&>::type in_plane(const Matrix<T,m>& U) {
  return U;
}

template<int d> static inline typename enable_if_c<d==2,Matrix<T,3,2>>::type in_plane(const Matrix<T,3>& U) {
  return Matrix<T,3,2>(U.column(0),U.column(1));
}

template<class TV,int d> void FiniteVolume<TV,d>::update_position(Array<const TV> X,bool definite_) {
  definite = definite_;
  stress_derivatives_valid = false;
  U.clear();
  U.resize(strain->elements.size(),uninit);
  De_inverse_hat.clear();
  De_inverse_hat.resize(strain->elements.size(),uninit);
  Fe_hat.clear();
  Fe_hat.resize(strain->elements.size(),uninit);
  V.clear();
  if (anisotropic)
    V.resize(strain->elements.size(),uninit);
  for (int t=0;t<strain->elements.size();t++) {
    Matrix<T,d> V_;
    if (plasticity) {
      Matrix<T,m,d> F = strain->F(X,t);
      (F*plasticity->Fp_inverse(t)).fast_singular_value_decomposition(U[t],Fe_hat[t],V_);
      DiagonalMatrix<T,d> Fe_project_hat;
      if (plasticity->project_Fe(Fe_hat[t],Fe_project_hat)) {
        plasticity->project_Fp(t,Fe_project_hat.inverse()*in_plane<d>(U[t]).transpose_times(F));
        (F*plasticity->Fp_inverse[t]).fast_singular_value_decomposition(U[t],Fe_hat[t],V_);
      }
      De_inverse_hat[t] = strain->Dm_inverse[t]*plasticity->Fp_inverse[t]*V_;
      Be_scales[t] = -(T)1/Factorial<d>::value/De_inverse_hat[t].determinant();
    } else {
      strain->F(X,t).fast_singular_value_decomposition(U[t],Fe_hat[t],V_);
      De_inverse_hat[t] = strain->Dm_inverse[t]*V_;
    }
    if (anisotropic) anisotropic->update_position(Fe_hat[t],V_,t);
    else isotropic->update_position(Fe_hat[t],t);
    if (anisotropic) V[t] = V_;
  }
}

template<class TV,int d> typename TV::Scalar FiniteVolume<TV,d>::elastic_energy() const {
  T energy = 0;
  if (anisotropic)
    for (int t=0;t<strain->elements.size();t++)
      energy -= Be_scales[t]*anisotropic->elastic_energy(Fe_hat[t],V[t],t);
  else
    for (int t=0;t<strain->elements.size();t++)
      energy -= Be_scales[t]*isotropic->elastic_energy(Fe_hat[t],t);
  return energy;
}

template<class TV,int d> void FiniteVolume<TV,d>::add_elastic_force(RawArray<TV> F) const {
  if (anisotropic)
    for (int t=0;t<strain->elements.size();t++) {
      Matrix<T,m,d> forces = in_plane<d>(U[t])*anisotropic->P_From_Strain(Fe_hat[t],V[t],Be_scales[t],t).times_transpose(De_inverse_hat[t]);
      strain->distribute_force(F,t,forces);
    }
  else
    for (int t=0;t<strain->elements.size();t++) {
      Matrix<T,m,d> forces = in_plane<d>(U[t])*isotropic->P_From_Strain(Fe_hat[t],Be_scales[t],t).times_transpose(De_inverse_hat[t]);
      strain->distribute_force(F,t,forces);
    }
}

template<int m,int d> static inline typename enable_if_c<m==d,const DiagonalizedIsotropicStressDerivative<T,m>&>::type
add_out_of_plane(const IsotropicConstitutiveModel<T,d>& model, const DiagonalMatrix<T,d>& F_hat, const DiagonalizedIsotropicStressDerivative<T,d>& in_plane, int t) {
  return in_plane;
}

template<int m> static inline typename enable_if_c<m==3,DiagonalizedIsotropicStressDerivative<T,3,2>>::type
add_out_of_plane(const IsotropicConstitutiveModel<T,2>& model, const DiagonalMatrix<T,2>& F_hat, const DiagonalizedIsotropicStressDerivative<T,2>& in_plane, int t) {
  DiagonalizedIsotropicStressDerivative<T,3,2> A;
  A.A = in_plane;
  DiagonalMatrix<T,2> P = model.P_From_Strain(F_hat,1,t), F_clamp = model.clamp_f(F_hat);
  A.x2020 = P.x00/F_clamp.x00;
  A.x2121 = P.x11/F_clamp.x11;
  return A;
}

template<class TV,int d> void FiniteVolume<TV,d>::update_stress_derivatives() const {
  if (stress_derivatives_valid) return;
  GEODE_ASSERT(isotropic || (int)TV::m==(int)d); // codimension zero only for anisotropic for now
  if (anisotropic && !anisotropic->use_isotropic_stress_derivative()) {
    dP_dFe.clear();
    dP_dFe.resize(strain->elements.size(),uninit);
    for (int t=0;t<strain->elements.size();t++) {
      dP_dFe[t] = anisotropic->stress_derivative(Fe_hat[t],V[t],t);
      if (definite) dP_dFe[t].enforce_definiteness();
    }
  } else {
    dPi_dFe.clear();
    dPi_dFe.resize(strain->elements.size(),uninit);
    for (int t=0;t<strain->elements.size();t++) {
      dPi_dFe[t] = add_out_of_plane<m>(*isotropic,Fe_hat[t],model->isotropic_stress_derivative(Fe_hat[t],t),t);
      if(definite) dPi_dFe[t].enforce_definiteness();
    }
  }
  stress_derivatives_valid = true;
}

template<class TV,int d> void FiniteVolume<TV,d>::add_elastic_differential(RawArray<TV> dF, RawArray<const TV> dX) const {
  update_stress_derivatives();
  if (anisotropic && !anisotropic->use_isotropic_stress_derivative())
    for (int t=0;t<strain->elements.size();t++) {
      Matrix<T,m,d> dDs = strain->Ds(dX,t),
                    Up = in_plane<d>(U[t]),
                    dG = Up*(Be_scales[t]*dP_dFe[t].differential(Up.transpose_times(dDs)*De_inverse_hat[t]).times_transpose(De_inverse_hat[t]));
      strain->distribute_force(dF,t,dG);
    }
  else
    for (int t=0;t<strain->elements.size();t++) {
      Matrix<T,m,d> dDs = strain->Ds(dX,t),
                    dG = U[t]*(Be_scales[t]*dPi_dFe[t].differential(U[t].transpose_times(dDs)*De_inverse_hat[t]).times_transpose(De_inverse_hat[t]));
      strain->distribute_force(dF,t,dG);
    }
}

template<class TV,int d> void FiniteVolume<TV,d>::add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,m>> dFdX) const {
  update_stress_derivatives();
  if (anisotropic && !anisotropic->use_isotropic_stress_derivative())
    GEODE_NOT_IMPLEMENTED();
  else {
    Matrix<T,m> dGdD[d][d];
    for (int t=0;t<strain->elements.size();t++) {
      for (int i=0;i<d;i++)
        for(int j=0;j<m;j++) {
          Matrix<T,m,d> dDs;
          dDs(j,i) = 1;
          Matrix<T,m,d> dG = U[t]*(Be_scales[t]*dPi_dFe[t].differential(U[t].transpose_times(dDs)*De_inverse_hat[t]).times_transpose(De_inverse_hat[t]));
          for (int k=0;k<d;k++)
            dGdD[k][i].set_column(j,dG.column(k));
        }
      Vector<int,d+1> nodes = strain->elements[t];
      for (int i=0;i<d;i++)
        dFdX[nodes[i+1]] += assume_symmetric(dGdD[i][i]);
      SymmetricMatrix<T,m> sum;
      for (int i=0;i<d;i++)
        for (int j=0;j<d;j++)
          sum += assume_symmetric(dGdD[i][j]);
      dFdX[nodes[0]] += sum;
    }
  }
}

template<class TV,int d> void FiniteVolume<TV,d>::add_elastic_gradient(SolidMatrix<TV>& matrix) const {
  update_stress_derivatives();
  if (anisotropic && !anisotropic->use_isotropic_stress_derivative())
    GEODE_NOT_IMPLEMENTED();
  else {
    Matrix<T,m> dGdD[d+1][d+1];
    for (int t=0;t<strain->elements.size();t++) {
      for (int i=0;i<d;i++)
        for (int j=0;j<m;j++) {
          Matrix<T,m,d> dDs;
          dDs(j,i) = 1;
          Matrix<T,m,d> dG = U[t]*(Be_scales[t]*dPi_dFe[t].differential(U[t].transpose_times(dDs)*De_inverse_hat[t]).times_transpose(De_inverse_hat[t]));
          for (int k=0;k<d;k++)
            dGdD[k+1][i+1].set_column(j,dG.column(k));
        }
      Matrix<T,m> sum;
      for (int i=0;i<d;i++) {
        Matrix<T,m> sum_i;
        for (int j=0;j<d;j++)
          sum_i -= dGdD[i+1][j+1];
        dGdD[i+1][0] = sum_i;
        sum -= sum_i;
      }
      dGdD[0][0] = sum;
      Vector<int,d+1> nodes = strain->elements[t];
      for (int j=0;j<d+1;j++)
        for (int i=j;i<d+1;i++)
          matrix.add_entry(nodes[i],nodes[j],dGdD[i][j]);
    }
  }
}

template<class TV,int d> typename TV::Scalar FiniteVolume<TV,d>::damping_energy(RawArray<const TV> V) const {
  T energy = 0;
  for (int t=0;t<strain->elements.size();t++) {
    Matrix<T,d> Fe_dot_hat = in_plane<d>(U[t]).transpose_times(strain->Ds(V,t))*De_inverse_hat[t];
    energy -= Be_scales[t]*model->damping_energy(Fe_hat[t],Fe_dot_hat,t);
  }
  return energy;
}

template<class TV,int d> void FiniteVolume<TV,d>::add_damping_force(RawArray<TV> F,RawArray<const TV> V) const {
  for (int t=0;t<strain->elements.size();t++) {
    Matrix<T,m,d> Up = in_plane<d>(U[t]);
    Matrix<T,d> Fe_dot_hat = Up.transpose_times(strain->Ds(V,t))*De_inverse_hat[t];
    Matrix<T,m,d> forces = Up*model->P_From_Strain_Rate(Fe_hat[t],Fe_dot_hat,Be_scales[t],t).times_transpose(De_inverse_hat[t]);
    strain->distribute_force(F,t,forces);
  }
}

template<class TV,int d> void FiniteVolume<TV,d>::add_damping_gradient(SolidMatrix<TV>& matrix) const {
  Matrix<T,m> dGdD[d+1][d+1];
  for (int t=0;t<strain->elements.size();t++) {
    Matrix<T,m,d> Up = in_plane<d>(U[t]);
    for (int i=0;i<d;i++)
      for (int j=0;j<m;j++) {
        Matrix<T,m,d> Ds_dot;
        Ds_dot(j,i) = 1;
        Matrix<T,d> Fe_dot_hat = Up.transpose_times(Ds_dot)*De_inverse_hat[t];
        Matrix<T,m,d> dG = Up*model->P_From_Strain_Rate(Fe_hat[t],Fe_dot_hat,Be_scales[t],t).times_transpose(De_inverse_hat[t]);
        for (int k=0;k<d;k++)
          dGdD[k+1][i+1].set_column(j,dG.column(k));
      }
    Matrix<T,m> sum;
    for (int i=0;i<d;i++) {
      Matrix<T,m> sum_i;
      for (int j=0;j<d;j++)
        sum_i -= dGdD[i+1][j+1];
      dGdD[i+1][0] = sum_i;
      sum -= sum_i;
    }
    dGdD[0][0] = sum;
    Vector<int,d+1> nodes = strain->elements[t];
    for (int j=0;j<d+1;j++)
      for (int i=j;i<d+1;i++)
        matrix.add_entry(nodes[i],nodes[j],dGdD[i][j]);
  }
}

template<class TV,int d> void FiniteVolume<TV,d>::add_frequency_squared(RawArray<T> frequency_squared) const {
  Hashtable<int,T> particle_frequency_squared;
  for (int t=0;t<strain->elements.size();t++) {
    T elastic_squared = model->maximum_elastic_stiffness(t)/(sqr(strain->rest_altitude(t))*density);
    const Vector<int,d+1>& nodes = strain->elements(t);
    for (int j=0;j<nodes.m;j++) {
      T& data = particle_frequency_squared.get_or_insert(nodes[j]);
      data = max(data,elastic_squared);
    }
  }
  for (auto& it : particle_frequency_squared)
    frequency_squared[it.key()] += it.data();
}

template<class TV,int d> typename TV::Scalar FiniteVolume<TV,d>::strain_rate(RawArray<const TV> V) const {
  T strain_rate = 0;
  for (int t=0;t<strain->elements.size();t++)
    strain_rate = max(strain_rate,strain->F(V,t).maxabs());
  return strain_rate;
}

template class FiniteVolume<Vector<T,2>,2>;
template class FiniteVolume<Vector<T,3>,2>;
template class FiniteVolume<Vector<T,3>,3>;
}
using namespace geode;

template<int m,int d> static void wrap_helper() {
  typedef FiniteVolume<Vector<T,m>,d> Self;
  static const string name = format("FiniteVolume%s",d==3?"3d":m==3?"S3d":"2d");
  Class<Self>(name.c_str())
    .GEODE_INIT(StrainMeasure<T,d>&,T,ConstitutiveModel<T,d>&,Ptr<PlasticityModel<T,d>>)
    ;
}

void wrap_finite_volume() {
  wrap_helper<2,2>();
  wrap_helper<3,2>();
  wrap_helper<3,3>();
}
