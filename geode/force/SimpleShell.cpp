// Specialized finite volume model for in-plane, anisotropic shell forces.

#include <geode/force/SimpleShell.h>
#include <geode/force/StrainMeasure.h>
#include <geode/utility/Log.h>
#include <geode/vector/normalize.h>
#include <geode/vector/DiagonalMatrix.h>
#include <geode/vector/SolidMatrix.h>
#include <geode/vector/UpperTriangularMatrix.h>
namespace geode {

using Log::cout;
using std::endl;
typedef real T;
typedef Vector<T,3> TV;
typedef SymmetricMatrix<T,2> SM2;
typedef StrainMeasure<T,2> Strain;

SimpleShell::SimpleShell(const TriangleSoup& mesh, RawArray<const Matrix<T,2>> Dm, const T density)
  : density(density)
  , shear_stiffness(0)
  , F_threshold(.1)
  , nodes_(mesh.nodes())
  , definite_(false)
  , info(mesh.elements.size(),uninit) {
  GEODE_ASSERT(mesh.elements.size()==Dm.size());
  for (int t=0;t<mesh.elements.size();t++) {
    auto& I = info[t];
    I.nodes = mesh.elements[t];
    const auto det = Dm[t].determinant();
    if (det <= 0)
      throw RuntimeError("SimpleShell: Inverted or degenerate rest state");
    I.inv_Dm = Dm[t].inverse();
    I.scale = -(T)1/2*det;
  }
}

SimpleShell::~SimpleShell() {}

int SimpleShell::nodes() const {
  return nodes_;
}

void SimpleShell::structure(SolidMatrixStructure& structure) const {
  for (const auto& I : info)
    for (int i=0;i<I.nodes.size();i++)
      for (int j=i+1;j<I.nodes.size();j++)
        structure.add_entry(I.nodes[i],I.nodes[j]);
}

void SimpleShell::update_position(Array<const TV> X_, bool definite) {
  GEODE_ASSERT(X_.size() >= nodes_);
  X = X_;
  definite_ = definite;
  if (definite)
    update_position_helper<true>();
  else
    update_position_helper<false>();
}

template<bool definite> void SimpleShell::update_position_helper() {
  for (auto& I : info) {
    // Rotate F to be symmetric
    (Strain::Ds(X,I.nodes)*I.inv_Dm).fast_indefinite_polar_decomposition(I.Q,I.Fh);

    // Precompute as much force differential information as we can.
    // First, evaluate force and force differential pretending that Fh stays symmetric
    auto Phs = simple_P(I);
    // DPhs is the Hessian density tensor mapping symmetric 2x2 changes in Fh to symmetric 2x2 force differentials.  However,
    // what we actually need is the 6x6 (or (3x2)x(3x2)) tensor mapping any change in Fh to a force differential, including
    // out of plane and 2x2 antisymmetric components.  As derived in simple-shell.nb, this tensor breaks up into two pieces,
    // one 4x4 diagonal block mapping in-plane to in-plane, and one 2x2 diagonal block mapping out-of-plane to out-of-plane.
    // The tensor also breaks into two components via the product rule, one involving DPhs which is entirely in-plane, and
    // one involving the rotation of Phs which is both in and out of plane.  The DPhs-related term is automatically positive
    // semidefinite assuming DPhs itself is, but the Phs term may not be.  Happily, although the Phs term is a 4x4 matrix,
    // only one row and one column are nonzero, so we have an easy closed form eigensystem and easy definiteness projection.
    // Finally, we evaluate the Phs component using Fh clamped away from zero to avoid singularities.
    //
    // First, the 2x2 out-of-plane block:
    const auto Fhc = F_threshold+(I.Fh-F_threshold).positive_definite_part();
    const T ctrace = Fhc.trace(),
            det = Fhc.determinant();
    I.H_nonplanar = 1/(ctrace*det)*( Phs.x00*SM2(det+sqr(Fhc.x11),-Fhc.x10*Fhc.x11,sqr(Fhc.x10))
                                    +Phs.x11*SM2(sqr(Fhc.x10),-Fhc.x10*Fhc.x00,det+sqr(Fhc.x00))
                                    +Phs.x10*SM2(-Fhc.x10*Fhc.x11,Fhc.x00*Fhc.x11,-Fhc.x10*Fhc.x00));
    if (definite)
      I.H_nonplanar = -(-I.H_nonplanar).positive_definite_part();

    // Next, the 4x4 in-plane block due to Phs.  Only the row and column of the 2x2 antisymmetric component are nonzero,
    // which amounts to four nonzero entries.  In simple-shell.nb, H_planar is {a,b,c,d}.
    I.H_planar = 1/sqr(ctrace)*vec(sqrt(2)*Fhc.x10*(Phs.x00-Phs.x11)+sqrt(2)*Fhc.x11*Phs.x10,
                                   ctrace*(Phs.x11-Phs.x00),
                                   Fhc.x11*Phs.x00+Fhc.x00*Phs.x11-Fhc.x10*Phs.x10,
                                   sqrt(2)*Fhc.x10*(Phs.x00-Phs.x11)-sqrt(2)*Fhc.x00*Phs.x10);
    // H_planar represents the matrix with one nonzero row plus one nonzero column, both given by the vector H_planar
    // (thus there is a factor of two where the row and column intersect).  Such matrices are rank two with one negative
    // and one positive eigenvalue.  If we're performing definiteness projection, we keep only the negative eigenvalue.
    if (definite) {
      const T norm = magnitude(I.H_planar),
              a_neg = I.H_planar.z-norm;
      auto u_neg = vec(I.H_planar.x,I.H_planar.y,a_neg,I.H_planar.w);
      const T sqr_neg = sqr_magnitude(u_neg);
      u_neg *= sqr_neg?sqrt(-a_neg/sqr_neg):0;
      I.H_planar = u_neg;
    }

    // Finally, the 4x4 in-plane block due to DPhs
    const T trace = I.Fh.trace(),
            inv_trace = trace ? 1/trace : 0;
    I.c0 = sqrt(2)*I.Fh.x10*inv_trace,
    I.c1 = (I.Fh.x00-I.Fh.x11)*inv_trace;
  }
}

SM2 SimpleShell::stiffness() const {
  return SM2(stretch_stiffness.x,
             shear_stiffness,
             stretch_stiffness.y);
}

// Set to true to switch energy from quadratic to linear for testing purposes
static const bool tweak = false;

T SimpleShell::elastic_energy() const {
  if (tweak)
    GEODE_WARNING("Linear shell energy enabled: use for debugging purposes only");
  T energy = 0;
  const auto stiff = stiffness();
  for (const auto& I : info)
    energy -= !tweak ? I.scale*( stiff.x00*sqr(I.Fh.x00-1)
                                +stiff.x10*sqr(I.Fh.x10)
                                +stiff.x11*sqr(I.Fh.x11-1))
                     : I.scale*( stiff.x00*I.Fh.x00
                                +stiff.x10*I.Fh.x10
                                +stiff.x11*I.Fh.x11);
  return !tweak ? energy/2
                : energy;
}

inline SM2 SimpleShell::simple_P(const Info& I) const {
  const auto stiff = I.scale*stiffness();
  return !tweak ? SM2(stiff.x00*(I.Fh.x00-1),
                      stiff.x10* I.Fh.x10,
                      stiff.x11*(I.Fh.x11-1))
                : stiff;
}

template<bool definite> inline SM2 SimpleShell::simple_DP(const Info& I) const {
  // Both of these cases are unconditionally definite by default
  return !tweak ? I.scale*stiffness()
                : SM2();
}

static inline Matrix<T,3,2> in_plane(const Matrix<T,3>& Q) {
  return Matrix<T,3,2>(Q.column(0),Q.column(1));
}

void SimpleShell::add_elastic_force(RawArray<TV> F) const {
  for (const auto& I : info) {
    // Evaluate force pretending that Fh stays symmetric
    const auto Phs = simple_P(I);
    // Account for rotation induced by antisymmetric components of d(Q'F).
    // These formulas are stable by positive semidefiniteness of Fh.
    const T ds = I.Fh.x00+I.Fh.x11,
            inv_ds = ds ? 1/ds : 0;
    const T go = inv_ds*(I.Fh.x10*(Phs.x11-Phs.x00)+(T).5*(I.Fh.x00-I.Fh.x11)*Phs.x10);
    const Matrix<T,2> Ph(Phs.x00,
                         (T).5*Phs.x10-go,
                         (T).5*Phs.x10+go,
                         Phs.x11);
    // Apply force
    const auto forces = in_plane(I.Q)*Ph.times_transpose(I.inv_Dm);
    Strain::distribute_force(F,I.nodes,forces);
  }
}

template<bool definite> inline Matrix<T,3,2> SimpleShell::force_differential(const Info& I, const Matrix<T,3,2>& dDs) const {
  // Rotate dF into polar decomposition space.  Note that dFh is 3x2, not 2x2, since it includes out of plane terms.
  // Then warp fxy,fyx into (scaled) symmetric and nonsymmetric part, respectively
  Matrix<T,3,2> dFh = I.Q.transpose_times(dDs*I.inv_Dm);
  const T dFh_sym = sqrt(.5)*(dFh(0,1)+dFh(1,0));
  dFh(1,0) = sqrt(.5)*(dFh(0,1)-dFh(1,0));
  dFh(0,1) = dFh_sym;
  // Evaluate force and force differential pretending that Fh stays symmetric
  auto DPhs = simple_DP<definite>(I);
  // First, the 2x2 out-of-plane block:
  const auto dP_nonplanar = I.H_nonplanar*vec(dFh(2,0),dFh(2,1));
  // Next, the 4x4 in-plane block due to Phs
  Vector<T,4> dP_planar;
  const auto dF_planar = vec(dFh(0,0),dFh(0,1),dFh(1,0),dFh(1,1));
  if (definite)
    dP_planar = -dot(dF_planar,I.H_planar)*I.H_planar;
  else {
    dP_planar = I.H_planar*dF_planar.z;
    dP_planar.z += dot(I.H_planar,dF_planar);
  }
  // Finally, the 4x4 in-plane block due to DPhs
  const T m00 = DPhs.x00*(dFh(0,0)-I.c0*dFh(1,0)),
          m11 = DPhs.x11*(dFh(1,1)+I.c0*dFh(1,0)),
          m01 = .5*DPhs.x10*(dFh(0,1)+I.c1*dFh(1,0));
  dP_planar += vec(m00,m01,I.c1*m01+I.c0*(m11-m00),m11);
  // Unwarp and assemble into dP
  dP_planar = vec(dP_planar.x,sqrt(.5)*(dP_planar.y+dP_planar.z),
                              sqrt(.5)*(dP_planar.y-dP_planar.z),dP_planar.w);
  const Matrix<T,3,2> dPh(dP_planar.x,dP_planar.z,dP_nonplanar.x,
                          dP_planar.y,dP_planar.w,dP_nonplanar.y);
  // Unrotate force differential
  return I.Q*dPh.times_transpose(I.inv_Dm);
}

void SimpleShell::add_elastic_differential(RawArray<TV> dF, RawArray<const TV> dX) const {
  if (definite_)
    for (const auto& I : info)
      Strain::distribute_force(dF,I.nodes,force_differential<true >(I,Strain::Ds(dX,I.nodes)));
  else
    for (const auto& I : info)
      Strain::distribute_force(dF,I.nodes,force_differential<false>(I,Strain::Ds(dX,I.nodes)));
}

void SimpleShell::add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,3>> dFdX) const {
  GEODE_NOT_IMPLEMENTED();
}

template<bool definite> void SimpleShell::add_elastic_gradient_helper(SolidMatrix<TV>& matrix) const {
  const int m = 3, d = 2;
  Matrix<T,m> dGdD[d+1][d+1];
  for (const auto& I : info) {
    for (int i=0;i<d;i++)
      for (int j=0;j<m;j++) {
        Matrix<T,m,d> dDs;
        dDs(j,i) = 1;
        Matrix<T,m,d> dG = force_differential<definite>(I,dDs);
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
    for (int j=0;j<d+1;j++)
      for (int i=j;i<d+1;i++)
        matrix.add_entry(I.nodes[i],I.nodes[j],dGdD[i][j]);
  }
}

void SimpleShell::add_elastic_gradient(SolidMatrix<TV>& matrix) const {
  if (definite_)
    add_elastic_gradient_helper<true >(matrix);
  else
    add_elastic_gradient_helper<false>(matrix);
}

T SimpleShell::damping_energy(RawArray<const TV> V) const {
  return 0;
}

void SimpleShell::add_damping_force(RawArray<TV> F, RawArray<const TV> V) const {
  return;
}

void SimpleShell::add_damping_gradient(SolidMatrix<TV>& matrix) const {
  return;
}

void SimpleShell::add_frequency_squared(RawArray<T> frequency_squared) const {
  const T max_stiff = stiffness().maxabs();
  Hashtable<int,T> particle_frequency_squared;
  for (const auto& I : info) {
    const T elastic_squared = max_stiff/(sqr(I.inv_Dm.inverse().simplex_minimum_altitude())*density);
    for (const int n : I.nodes) {
      T& data = particle_frequency_squared.get_or_insert(n);
      data = max(data,elastic_squared);
    }
  }
  for (auto& it : particle_frequency_squared)
    frequency_squared[it.x] += it.y;
}

T SimpleShell::strain_rate(RawArray<const TV> V) const {
  T strain_rate = 0;
  for (const auto& I : info)
    strain_rate = max(strain_rate,(Strain::Ds(V,I.nodes)*I.inv_Dm).maxabs());
  return strain_rate;
}

}
