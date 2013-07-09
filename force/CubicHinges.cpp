// Cubic hinges based on Garg et al. 2007

#include <other/core/force/CubicHinges.h>
#include <other/core/array/view.h>
#include <other/core/math/copysign.h>
#include <other/core/geometry/Triangle3d.h>
#include <other/core/python/Class.h>
#include <other/core/utility/Log.h>
#include <other/core/vector/normalize.h>
#include <other/core/vector/SolidMatrix.h>
#include <other/core/vector/SymmetricMatrix.h>
namespace other {

using Log::cout;
using std::endl;
typedef real T;
typedef Vector<T,2> TV2;
typedef Vector<T,3> TV3;

template<> OTHER_DEFINE_TYPE(CubicHinges<TV2>)
template<> OTHER_DEFINE_TYPE(CubicHinges<TV3>)

// In both 2d and 3d cases, we want to approximate the nonlinear energy
//
//   E = 1/2 int_S (H - H0)^2 da
//
// where H is the mean curvature.  Here are two special cases we will use for verification (assuming flat rest angles):
//
//   E(r S_1) = 1/2 * 2pi r * (1/r)^2 = pi/r
//   E(r S_2) = 1/2 * 4pi r^2 * (1/r)^2 = 2pi
//
// We discretize this energy directly with the "slow_elastic_energy" function (used only for testing), then approximate
// it further assuming nearly isometric deformations for the rest of the class.  Our discretized slow energy is
//
//   E = k sum_b |e_b|^2/A_b (2sin((psi-psi_0)/2))^2
//
// where k = 1 in 2D and 3/2 in 3D.  Garg et al. approximate this by a sum of a quadratic dot product term and a determinant term
// the latter vanishing for zero rest angle.  Unfortunately, unlike in the original quadratic energy paper, the quadratic energy
// term in the cubic paper is indefinite.  Therefore, we use a slightly different form which preserves the definiteness of the
// quadratic part from the original paper.  If a and a0 are the current and rest angles, we have
//
//   4 sin^2 (a/2-a0/2) = 4 (sin a/2 cos a0/2 - cos a/2 sin a0/2)^2
//                      = 4 sin^2 a/2 cos^2 a0/2 - 8 sin a/2 cos a/2 sin a0/2 cos a0/2 + 4 cos^2 a/2 sin^2 a0/2
//                      = 4 sin^2 a0/2 + 4 sin^2 a/2 (cos^2 a0/2 - sin^2 a0/2) - 2 sin a sin a0
//                      = 4 sin^2 a0/2 + 4 sin^2 a/2 cos a0 - 2 sin a sin a0
//                      = 4 sin^2 a0/2 + cos a0/|e0|^2 |t0-t1|^2 - 2 sin a sin a0
//
// where t0,t1 are orthogonal to e0 in each triangle plane with |t0|=|t1|=|e0| (see Garg et al.).  Thus, if a0 = 0, we reduce
// to the positive definite quadratic energy from the original paper.  sin a is approximated as in Garg et al.:
//
//   sin a = cross(t0,t1) = e0 e1/(l0 l1) in 2D
//   sin a = -beta/|e0|^2 det(e0,e1,e2)   in 3D
//
// which reduces to the positive definite quadratic energy from the original paper in the flat case.  Thus our discrete energy is
//
//   E = k/A (4|e|^2 sin^2 a0/2 + cos a0 |Ax|^2 - 2|e|^2 sin a0 sin a
//   Ax = e1/l1 - e0/l0 = (1/l0)x0 - (1/l0+1/l1)x1 + (1/l1)x2                                                                  in 2D
//   Ax = cot03 e1 + cot01 e3 + cot04 e2 + cot02 e4 = (cot04+cot02)x0 + (-cot03-cot04)x1 + (-cot01-cot02)x2 + (cot03+cot01)x3  in 3D

template<> Array<T> CubicHinges<TV2>::angles(RawArray<const Vector<int,3>> bends, RawArray<const TV2> X) {
  if (bends.size())
    OTHER_ASSERT(X.size()>scalar_view(bends).max());
  Array<T> angles(bends.size(),false);
  for (int b=0;b<bends.size();b++) {
    int i0,i1,i2;bends[b].get(i0,i1,i2);
    const TV x0 = X[i0], x1 = X[i1], x2 = X[i2];
    angles[b] = angle_between(x1-x0,x2-x1);
  }
  return angles;
}

template<> Array<T> CubicHinges<TV3>::angles(RawArray<const Vector<int,4>> bends, RawArray<const TV3> X) {
  if (bends.size())
    OTHER_ASSERT(X.size()>scalar_view(bends).max());
  Array<T> angles(bends.size(),false);
  for (int b=0;b<bends.size();b++) {
    int i0,i1,i2,i3;bends[b].get(i0,i1,i2,i3);
    const TV x0 = X[i0], x1 = X[i1], x2 = X[i2], x3 = X[i3],
             n0 = normal(x0,x2,x1),
             n1 = normal(x1,x2,x3);
    angles[b] = copysign(acos(clamp(dot(n0,n1),-1.,1.)),dot(n1-n0,x3-x0));
  }
  return angles;
}

template<> T CubicHinges<TV2>::slow_elastic_energy(RawArray<const T> angles, RawArray<const TV> restX, RawArray<const TV> X) const {
  OTHER_ASSERT(X.size()>=nodes_);
  T sum = 0;
  const T scale = stiffness;
  for (int b=0;b<bends.size();b++) {
    int i0,i1,i2;bends[b].get(i0,i1,i2);
    const TV  x0 =     X[i0],  x1 =     X[i1],  x2 =     X[i2],
             rx0 = restX[i0], rx1 = restX[i1], rx2 = restX[i2];
    const T theta = angle_between(x1-x0,x2-x1);
    sum += scale/(magnitude(rx1-rx0)+magnitude(rx2-rx1))*sqr(2*sin((theta-angles[b])/2));
    // To check this formula, the energy of a radius r polygon with n segments is
    //
    //   E \approx n/2 * 2/(2*2pi r/n) * (2*sin(2pi/n/2))^2
    //     \approx n/2 * 2n/(4pi r) * (2pi/n)^2
    //     \approx 4pi^2/(4pi r) = pi/r
  }
  return sum;
}

template<> T CubicHinges<TV3>::slow_elastic_energy(RawArray<const T> angles, RawArray<const TV> restX, RawArray<const TV> X) const {
  OTHER_ASSERT(X.size()>=nodes_);
  T sum = 0;
  const T scale = 3./2*stiffness;
  for (int b=0;b<bends.size();b++) {
    int i0,i1,i2,i3;bends[b].get(i0,i1,i2,i3);
    const TV  x0 =     X[i0],  x1 =     X[i1],  x2 =     X[i2],  x3 =     X[i3],
             rx0 = restX[i0], rx1 = restX[i1], rx2 = restX[i2], rx3 = restX[i3];
    const Triangle<TV>  t0( x0, x2, x1),  t1( x1, x2, x3),
                       rt0(rx0,rx2,rx1), rt1(rx1,rx2,rx3);
    const T theta = copysign(acos(clamp(dot(t0.n,t1.n),-1.,1.)),dot(t1.n-t0.n,x3-x0));
    sum += scale*sqr_magnitude(rx2-rx1)/(rt0.area()+rt1.area())*sqr(2*sin((theta-angles[b])/2));
  }
  return sum;
}

static void compute_info(RawArray<const Vector<int,3>> bends, RawArray<const T> angles, RawArray<const TV2> X, RawArray<CubicHinges<TV2>::Info> info) {
  for (const int b : range(bends.size())) {
    auto& I = info[b];
    int i0,i1,i2;bends[b].get(i0,i1,i2);
    const TV2 x0 = X[i0], x1 = X[i1], x2 = X[i2];
    const T cos_rest = cos(angles[b]),
            sin_rest = sin(angles[b]),
            len0 = magnitude(x1-x0),
            len1 = magnitude(x2-x1),
            ratio = max(len0,len1)/min(len0,len1),
            decay = ratio<5?1:1/sqr(ratio), // Don't explode as much if edges of very different lengths are adjacent
            scale = 2*decay/(len0+len1);
    I.base = scale*(1-cos_rest);
    I.dot = scale*cos_rest;
    I.c = vec(1/len0,-1/len0-1/len1,1/len1);
    I.det = scale/(len0*len1)*sin_rest;
  }
}

static void compute_info(RawArray<const Vector<int,4>> bends, RawArray<const T> angles, RawArray<const TV3> X, RawArray<CubicHinges<TV3>::Info> info) {
  for (const int b : range(bends.size())) {
    auto& I = info[b];
    int i0,i1,i2,i3;bends[b].get(i0,i1,i2,i3);
    // Use our ordering for x, Garg et al.'s for e and t.
    const TV3 x0 = X[i0], x1 = X[i1], x2 = X[i2], x3 = X[i3],
              e0 = x2-x1,
              e1 = x3-x1,
              e2 = x0-x1,
              e3 = x3-x2,
              e4 = x0-x2;
    const T cross01 = magnitude(cross(e0,e1)),
            cross02 = magnitude(cross(e0,e2)),
            cross03 = cross01,
            cross04 = cross02,
            dot00 = sqr_magnitude(e0),
            dot01 =  dot(e0,e1),
            dot02 =  dot(e0,e2),
            dot03 = -dot(e0,e3),
            dot04 = -dot(e0,e4),
            cot01 = dot01/cross01,
            cot02 = dot02/cross02,
            cot03 = dot03/cross03,
            cot04 = dot04/cross04,
            max_cot = max(cot01,cot02,cot03,cot04),
            plate = 6/(cross01+cross02),
            beta = (cot01+cot03)*(cot02+cot04)/sqrt(dot00),
            cos_rest = cos(angles[b]),
            sin_rest = sin(angles[b]);
    // Degrade gracefully for bad triangles (those with max_cot > 5)
    const T scale = plate/sqr(max(1,max_cot/5));
    I.base = scale*dot00*(1-cos_rest);
    I.dot = scale*cos_rest;
    I.det = scale*beta*sin_rest;
    I.c.set(-cot02-cot04,cot03+cot04,cot01+cot02,-cot01-cot03); // Use our ordering for vertices
  }
}

template<class TV> CubicHinges<TV>::CubicHinges(Array<const Vector<int,d+2>> bends, RawArray<const T> angles, RawArray<const TV> X)
  : bends(bends)
  , stiffness(0)
  , damping(0)
  , simple_hessian(false)
  , nodes_(bends.size()?scalar_view(bends).max()+1:0)
  , info(bends.size()) {
  OTHER_ASSERT(bends.size()==angles.size());
  OTHER_ASSERT(X.size()>=nodes_);
  compute_info(bends,angles,X,info);
}

template<class TV> CubicHinges<TV>::~CubicHinges() {}

template<class TV> int CubicHinges<TV>::nodes() const {
  return nodes_;
}

template<class TV> void CubicHinges<TV>::update_position(Array<const TV> X_, bool definite) {
  // Not much to do since our stiffness matrix is so simple
  OTHER_ASSERT(X_.size()>=nodes_);
  if (definite && !simple_hessian)
    OTHER_NOT_IMPLEMENTED("Definiteness fix implemented only for the simple_hessian case");
  X = X_;
}

template<class TV> void CubicHinges<TV>::add_frequency_squared(RawArray<T> frequency_squared) const {
  // We assume edge forces are much stiffer than bending, so our CFL shouldn't matter
}

template<class TV> T CubicHinges<TV>::strain_rate(RawArray<const TV> V) const {
  // We assume edge forces are much stiffer than bending, so our CFL shouldn't matter
  return 0;
}

template<bool simple> static T energy_helper(RawArray<const Vector<int,3>> bends, RawArray<const CubicHinges<TV2>::Info> info, RawArray<const TV2> X) {
  T sum = 0;
  for (int b=0;b<bends.size();b++) {
    const auto& I = info[b];
    int i0,i1,i2;bends[b].get(i0,i1,i2);
    const TV2 x0 = X[i0], x1 = X[i1], x2 = X[i2],
              strain = I.c[0]*x0+I.c[1]*x1+I.c[2]*x2;
    sum += I.base+.5*I.dot*sqr_magnitude(strain);
    if (!simple)
      sum -= I.det*cross(x1-x0,x2-x1);
  }
  return sum;
}

template<bool simple> static T energy_helper(RawArray<const Vector<int,4>> bends, RawArray<const CubicHinges<TV3>::Info> info, RawArray<const TV3> X) {
  T sum = 0;
  for (int b=0;b<bends.size();b++) {
    const auto& I = info[b];
    int i0,i1,i2,i3;bends[b].get(i0,i1,i2,i3);
    const TV3 x0 = X[i0], x1 = X[i1], x2 = X[i2], x3 = X[i3],
              strain = I.c[0]*x0+I.c[1]*x1+I.c[2]*x2+I.c[3]*x3;
    sum += I.base+.5*I.dot*sqr_magnitude(strain);
    if (!simple)
      sum += I.det*det(x2-x1,x3-x1,x0-x1);
  }
  return sum;
}

template<class TV> T CubicHinges<TV>::elastic_energy() const {
  return stiffness?stiffness*energy_helper<false>(bends,info,X):0;
}

template<class TV> T CubicHinges<TV>::damping_energy(RawArray<const TV> V) const {
  return damping?damping*energy_helper<true>(bends,info,V):0;
}

template<bool simple> static void add_force_helper(RawArray<const Vector<int,3>> bends, RawArray<const CubicHinges<TV2>::Info> info, const T scale, RawArray<TV2> F, RawArray<const TV2> X) {
  if (!scale) return;
  for (int b=0;b<bends.size();b++) {
    const auto& I = info[b];
    int i0,i1,i2;bends[b].get(i0,i1,i2);
    const TV2 x0 = X[i0], x1 = X[i1], x2 = X[i2],
              stress = scale*I.dot*(I.c[0]*x0+I.c[1]*x1+I.c[2]*x2);
    TV2 f0 = I.c[0]*stress,
        f2 = I.c[2]*stress;
    if (!simple) {
      f0 -= scale*I.det*rotate_left_90(x2-x1);
      f2 -= scale*I.det*rotate_left_90(x1-x0);
    }
    F[i0] -= f0;
    F[i1] += f0+f2;
    F[i2] -= f2;
  }
}

template<bool simple> static void add_force_helper(RawArray<const Vector<int,4>> bends, RawArray<const CubicHinges<TV3>::Info> info, const T scale, RawArray<TV3> F, RawArray<const TV3> X) {
  if (!scale) return;
  for (int b=0;b<bends.size();b++) {
    const auto& I = info[b];
    int i0,i1,i2,i3;bends[b].get(i0,i1,i2,i3);
    const TV3 x0 = X[i0], x1 = X[i1], x2 = X[i2], x3 = X[i3],
              stress = scale*I.dot*(I.c[0]*x0+I.c[1]*x1+I.c[2]*x2+I.c[3]*x3);
    if (simple) {
      F[i0] -= I.c[0]*stress;
      F[i1] -= I.c[1]*stress;
      F[i2] -= I.c[2]*stress;
      F[i3] -= I.c[3]*stress;
    } else {
      const T cubic = scale*I.det;
      const TV3 x0 = X[i0], x1 = X[i1], x2 = X[i2], x3 = X[i3];
      const TV3 ce0 = cubic*(x2-x1),
                e1 = x3-x1,
                e2 = x0-x1,
                cross01 = cross(ce0,e1),
                cross12 = cubic*cross(e1,e2),
                cross20 = cross(e2,ce0);
      F[i0] -= I.c[0]*stress+cross01;
      F[i1] -= I.c[1]*stress-cross01-cross12-cross20;
      F[i2] -= I.c[2]*stress+cross12;
      F[i3] -= I.c[3]*stress+cross20;
    }
  }
}

template<class TV> void CubicHinges<TV>::add_elastic_force(RawArray<TV> F) const {
  OTHER_ASSERT(F.size()>=nodes_);
  add_force_helper<false>(bends,info,stiffness,F,X);
}

template<class TV> void CubicHinges<TV>::add_damping_force(RawArray<TV> F, RawArray<const TV> V) const {
  OTHER_ASSERT(F.size()>=nodes_ && V.size()>=nodes_);
  add_force_helper<true>(bends,info,damping,F,V);
}

template<> void CubicHinges<TV2>::add_elastic_differential(RawArray<TV> dF, RawArray<const TV> dX) const {
  // 2D forces are unconditionally linear, so we can always reuse force computation
  OTHER_ASSERT(dF.size()>=nodes_ && dX.size()>=nodes_);
  if (simple_hessian)
    add_force_helper<true>(bends,info,stiffness,dF,dX);
  else
    add_force_helper<false>(bends,info,stiffness,dF,dX);
}

template<> void CubicHinges<TV3>::add_elastic_differential(RawArray<TV> dF, RawArray<const TV> dX) const {
  OTHER_ASSERT(dF.size()>=nodes_ && dX.size()>=nodes_);
  if (simple_hessian) // In the simple case, the force is linear and the differential is easy
    add_force_helper<true>(bends,info,stiffness,dF,dX);
  else { // Otherwise, we need custom code
    OTHER_ASSERT(dF.size()>=nodes_ && dX.size()>=nodes_);
    const T scale = stiffness;
    if (!scale) return;
    RawArray<const TV> X = this->X;
    for (int b=0;b<bends.size();b++) {
      const auto& I = info[b];
      int i0,i1,i2,i3;bends[b].get(i0,i1,i2,i3);
      const TV x0 = X[i0], x1 = X[i1], x2 = X[i2], x3 = X[i3];
      const TV dx0 = dX[i0], dx1 = dX[i1], dx2 = dX[i2], dx3 = dX[i3];
      const TV dstress = scale*I.dot*(I.c[0]*dx0+I.c[1]*dx1+I.c[2]*dx2+I.c[3]*dx3);
      const T cubic = scale*I.det;
      const TV ce0 = cubic*(x2-x1),
               e1 = x3-x1,
               e2 = x0-x1,
               dce0 = cubic*(dx2-dx1),
               de1 = dx3-dx1,
               de2 = dx0-dx1,
               dcross01 = cross(ce0,de1)+cross(dce0,e1),
               dcross12 = cubic*(cross(e1,de2)+cross(de1,e2)),
               dcross20 = cross(e2,dce0)+cross(de2,ce0);
      dF[i0] -= I.c[0]*dstress+dcross01;
      dF[i1] -= I.c[1]*dstress-dcross01-dcross12-dcross20;
      dF[i2] -= I.c[2]*dstress+dcross12;
      dF[i3] -= I.c[3]*dstress+dcross20;
    }
  }
}

template<class TV> void CubicHinges<TV>::
structure(SolidMatrixStructure& structure) const {
  for (auto& bend : bends)
    for (int j=0;j<bend.size();j++)
      for (int i=0;i<j;i++)
        structure.add_entry(bend[i],bend[j]);
}

template<bool simple> static void add_gradient_helper(RawArray<const Vector<int,3>> bends, RawArray<const CubicHinges<TV2>::Info> info, const T scale, RawArray<const TV2> X, SolidMatrix<TV2>& matrix) {
  if (!scale) return;
  for (int b=0;b<bends.size();b++) {
    const auto& I = info[b];
    int i0,i1,i2;bends[b].get(i0,i1,i2);
    const T quad = -scale*I.dot;
    matrix.add_entry(i0,quad*sqr(I.c[0]));
    matrix.add_entry(i1,quad*sqr(I.c[1]));
    matrix.add_entry(i2,quad*sqr(I.c[2]));
    if (simple) {
      matrix.add_entry(i0,i1,quad*I.c[0]*I.c[1]);
      matrix.add_entry(i0,i2,quad*I.c[0]*I.c[2]);
      matrix.add_entry(i1,i2,quad*I.c[1]*I.c[2]);
    } else {
      const T a = scale*I.det;
      const Matrix<T,2> anti(0,a,-a,0); // x'Ay = cross(x,y), scaled
      matrix.add_entry(i0,i1,quad*I.c[0]*I.c[1]-anti);
      matrix.add_entry(i0,i2,quad*I.c[0]*I.c[2]+anti);
      matrix.add_entry(i1,i2,quad*I.c[1]*I.c[2]-anti);
    }
  }
}

template<bool simple> static void add_gradient_helper(RawArray<const Vector<int,4>> bends, RawArray<const CubicHinges<TV3>::Info> info, const T scale, RawArray<const TV3> X, SolidMatrix<TV3>& matrix) {
  if (!scale) return;
  for (int b=0;b<bends.size();b++) {
    const auto& I = info[b];
    int i0,i1,i2,i3;bends[b].get(i0,i1,i2,i3);
    const T quad = -scale*I.dot;
    matrix.add_entry(i0,quad*sqr(I.c[0]));
    matrix.add_entry(i1,quad*sqr(I.c[1]));
    matrix.add_entry(i2,quad*sqr(I.c[2]));
    matrix.add_entry(i3,quad*sqr(I.c[3]));
    if (simple) {
      matrix.add_entry(i0,i1,quad*I.c[0]*I.c[1]);
      matrix.add_entry(i0,i2,quad*I.c[0]*I.c[2]);
      matrix.add_entry(i0,i3,quad*I.c[0]*I.c[3]);
      matrix.add_entry(i1,i2,quad*I.c[1]*I.c[2]);
      matrix.add_entry(i1,i3,quad*I.c[1]*I.c[3]);
      matrix.add_entry(i2,i3,quad*I.c[2]*I.c[3]);
    } else {
      const T cubic = -scale*I.det;
      const TV3 x0 = cubic*X[i0], x1 = cubic*X[i1], x2 = cubic*X[i2], x3 = cubic*X[i3];
      matrix.add_entry(i0,i1,quad*I.c[0]*I.c[1]+cross_product_matrix(x3-x2)); //  e3
      matrix.add_entry(i0,i2,quad*I.c[0]*I.c[2]+cross_product_matrix(x1-x3)); // -e1
      matrix.add_entry(i0,i3,quad*I.c[0]*I.c[3]+cross_product_matrix(x2-x1)); //  e0
      matrix.add_entry(i1,i2,quad*I.c[1]*I.c[2]+cross_product_matrix(x3-x0));
      matrix.add_entry(i1,i3,quad*I.c[1]*I.c[3]+cross_product_matrix(x0-x2)); //  e4
      matrix.add_entry(i2,i3,quad*I.c[2]*I.c[3]+cross_product_matrix(x1-x0)); // -e2
    }
  }
}

template<class TV> void CubicHinges<TV>::
add_elastic_gradient(SolidMatrix<TV>& matrix) const {
  OTHER_ASSERT(matrix.size()>=nodes_);
  if (simple_hessian)
    add_gradient_helper<true>(bends,info,stiffness,X,matrix);
  else
    add_gradient_helper<false>(bends,info,stiffness,X,matrix);
}

template<class TV> void CubicHinges<TV>::
add_damping_gradient(SolidMatrix<TV>& matrix) const {
  OTHER_ASSERT(matrix.size()>=nodes_);
  add_gradient_helper<true>(bends,info,damping,X,matrix);
}

template<class TV> void CubicHinges<TV>::add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,d+1>> dFdX) const {
  OTHER_ASSERT(dFdX.size()>=nodes_);
  if (!stiffness) return;
  const T scale = stiffness;
  for (int b=0;b<bends.size();b++) {
    const auto bend = bends[b];
    const auto& I = info[b];
    const T quad = scale*I.dot;
    for (int i=0;i<bend.size();i++)
      dFdX[bend[i]] -= quad*sqr(I.c[i]);
  }
}

template class CubicHinges<TV2>;
template class CubicHinges<TV3>;
}
using namespace other;

template<int d> static void wrap_helper() {
  typedef Vector<T,d+1> TV;
  typedef CubicHinges<TV> Self;
  Class<Self>(d==1?"CubicHinges2d":"CubicHinges3d")
    .OTHER_INIT(Array<const Vector<int,d+2>>,RawArray<const T>,RawArray<const TV>)
    .OTHER_METHOD(slow_elastic_energy)
    .OTHER_METHOD(angles)
    .OTHER_FIELD(stiffness)
    .OTHER_FIELD(damping)
    .OTHER_FIELD(simple_hessian)
    ;
}

void wrap_cubic_hinges() {
  wrap_helper<1>();
  wrap_helper<2>();
}
