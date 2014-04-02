//#####################################################################
// Class StrainMeasure
//#####################################################################
#pragma once

#include <geode/array/Array.h>
#include <geode/vector/Vector.h>
#include <geode/utility/debug.h>
namespace geode {

template<class T,int d> class StrainMeasure : public Object {
public:
  GEODE_NEW_FRIEND
  typedef Object Base;

  const int nodes;
  const Array<const Vector<int,d+1>> elements;
  const Array<UpperTriangularMatrix<T,d>> Dm_inverse;

protected:
  // Must have d <= X.n <= 3
  StrainMeasure(Array<const Vector<int,d+1>> elements, RawArray<const T,2> X);
public:
  ~StrainMeasure();

  template<class TX> Matrix<T,TX::value_type::m,d> F(const TX& X, const int simplex) const {
    return Ds(X,simplex)*Dm_inverse(simplex);
  }

  template<class TX> T J(const TX& X, const int simplex) const {
    return Ds(X,simplex).parallelpiped_measure()*Dm_inverse(simplex).determinant();
  }

  T rest_altitude(const int simplex) const {
    return Dm_inverse(simplex).inverse().simplex_minimum_altitude();
  }

  template<class TX> Matrix<T,TX::value_type::m,d> Ds(const TX& X, const int simplex) const {
    return Ds(X,elements[simplex]);
  }

  template<class TX> static Matrix<T,TX::value_type::m,d> Ds(const TX& X, const Vector<int,2>& nodes) { int i,j;     nodes.get(i,j);     return Matrix<T,TX::value_type::m,d>(X(j)-X(i)); }
  template<class TX> static Matrix<T,TX::value_type::m,d> Ds(const TX& X, const Vector<int,3>& nodes) { int i,j,k;   nodes.get(i,j,k);   return Matrix<T,TX::value_type::m,d>(X(j)-X(i),X(k)-X(i)); }
  template<class TX> static Matrix<T,TX::value_type::m,d> Ds(const TX& X, const Vector<int,4>& nodes) { int i,j,k,l; nodes.get(i,j,k,l); return Matrix<T,TX::value_type::m,d>(X(j)-X(i),X(k)-X(i),X(l)-X(i)); }

  template<class TX> void distribute_force(TX& F, const int element, const Matrix<T,TX::value_type::m,d>& forces) const {
    distribute_force(F,elements[element],forces);
  }

  template<class TX> static void distribute_force(TX& F, const Vector<int,3>& nodes, const Matrix<T,TX::value_type::m,2>& forces) {
    int i,j,k;nodes.get(i,j,k);
    F(i) -= forces.column(0)+forces.column(1);
    F(j) += forces.column(0);
    F(k) += forces.column(1);
  }

  template<class TX> static void distribute_force(TX& F, const Vector<int,4>& nodes, const Matrix<T,TX::value_type::m,3>& forces) {
    int i,j,k,l;nodes.get(i,j,k,l);
    F(i) -= forces.column(0)+forces.column(1)+forces.column(2);
    F(j) += forces.column(0);
    F(k) += forces.column(1);
    F(l) += forces.column(2);
  }

  T minimum_rest_altitude() const;
  void initialize_rest_state_to_equilateral(const T side_length);
  void print_altitude_statistics();
};

}
