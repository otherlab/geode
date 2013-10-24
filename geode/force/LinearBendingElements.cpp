#include <geode/force/LinearBendingElements.h>
#include <geode/mesh/SegmentMesh.h>
#include <geode/mesh/TriangleSoup.h>
#include <geode/utility/Log.h>
#include <geode/vector/SparseMatrix.h>
#include <geode/vector/SolidMatrix.h>
#include <geode/vector/SymmetricMatrix.h>
#include <geode/python/Class.h>
namespace geode {

typedef real T;
using Log::cout;
using std::endl;

template<> GEODE_DEFINE_TYPE(LinearBendingElements<Vector<T,2>>)
template<> GEODE_DEFINE_TYPE(LinearBendingElements<Vector<T,3>>)

template<class TV> static Ref<SparseMatrix> matrix_helper(const SegmentMesh& mesh,Array<const TV> X) {
  GEODE_ASSERT(mesh.nodes()<=X.size());
  Nested<const int> mesh_neighbors = mesh.neighbors();
  Hashtable<Vector<int,2>,T> entries;
  for(int p=0;p<mesh.nodes();p++) {
    RawArray<const int> neighbors = mesh_neighbors[p];
    for (int i=0;i<neighbors.size();i++) for(int j=i+1;j<neighbors.size();j++) {
      const Vector<int,3> nodes(neighbors[i],p,neighbors[j]);
      TV X0=X[nodes[0]],X1=X[nodes[1]],X2=X[nodes[2]];
      TV e01=X1-X0,e12=X2-X1;
      T length01=magnitude(e01),length12=magnitude(e12);
      // If edge lengths remain constant, we have |sin psi/2| = |(X0-X1)/length01 + (X2-X1)/length12|.
      // We'll define energy as stiffness/(2*length_scale)*(sin psi/2)^2
      T scale=length01+length12;
      Vector<T,3> c(1/length01,-1/length01-1/length12,1/length12);
      for(int i=0;i<nodes.m;i++) for(int j=i;j<nodes.m;j++)
        entries.get_or_insert(vec(nodes[i],nodes[j]).sorted())+=scale*c[i]*c[j];
    }
  }
  return new_<SparseMatrix>(entries);
}

template<class TV> static Ref<SparseMatrix> matrix_helper(const TriangleSoup& mesh,Array<const TV> X) {
  GEODE_ASSERT(mesh.nodes()<=X.size());
  Array<const Vector<int,4>> quadruples = mesh.bending_tuples();
  Hashtable<Vector<int,2>,T> entries;
  // Compute stiffness matrix.
  // For details see Wardetzky et al., "Discrete Quadratic Curvature energies", Computer Aided Geometric design, 2007.
  // Note that this computation depends only on the lengths of edges in the mesh, not the bend angles between triangles,
  // reflecting the fact that this class assumes a flat bending rest shape.
  for (Vector<int,4> nodes : mesh.bending_tuples()) {
    TV X0=X[nodes[0]],X1=X[nodes[1]],X2=X[nodes[2]],X3=X[nodes[3]];
    TV e0=X2-X1,e1=X3-X1,e2=X0-X1,e3=X3-X2,e4=X0-X2; // edge numbering matches Wardetzky et al. p. 16
    T cross01=magnitude(cross(e0,e1)),cross02=magnitude(cross(e0,e2)),
      cross03=magnitude(cross(e0,e3)),cross04=magnitude(cross(e0,e4));
    T dot01=dot(e0,e1),dot02=dot(e0,e2),dot03=-dot(e0,e3),dot04=-dot(e0,e4);
    T cot01=dot01/cross01,cot02=dot02/cross02,cot03=dot03/cross03,cot04=dot04/cross04;
    T max_cot=max(cot01,cot02,cot03,cot04);
    T scale=3/(cross01+cross02);
    if(max_cot>5) scale*=25/sqr(max_cot);
    Vector<T,4> c;c[1]=cot03+cot04;c[2]=cot01+cot02;c[3]=-cot01-cot03;c[0]=-cot02-cot04; // node numbering matches bending quadruple
    for(int i=0;i<nodes.m;i++) for(int j=i;j<nodes.m;j++)
      entries.get_or_insert(vec(nodes[i],nodes[j]).sorted())+=scale*c[i]*c[j];
  }
  return new_<SparseMatrix>(entries);
}

template<class TV> LinearBendingElements<TV>::LinearBendingElements(const Mesh& mesh,Array<const TV> X)
  : mesh(ref(mesh))
  , stiffness(0)
  , damping(0)
  , A(matrix_helper(mesh,X))
  , X(X) {
  // Print max diagonal element
  T max_diagonal = 0;
  for (int i=0;i<mesh.nodes();i++)
    if (A->A.size(i))
      max_diagonal = max(max_diagonal,A->A(i,0));
  cout<<"max diagonal element = "<<max_diagonal<<endl;

  // Verify that all edges occur in a triple/quadruple
  for (Vector<int,2> nodes : mesh.segment_mesh()->elements) {
    nodes = nodes.sorted();
    if (!A->contains_entry(nodes[0],nodes[1]))
      GEODE_FATAL_ERROR("Not all edges occur in bending quadruples");
  }
}

template<class TV> LinearBendingElements<TV>::~LinearBendingElements() {}

template<class TV> int LinearBendingElements<TV>::nodes() const {
  return mesh->nodes();
}

template<class TV> void LinearBendingElements<TV>::update_position(Array<const TV> X_,bool definite) {
  // Not much to do since our stiffness matrix is constant
  GEODE_ASSERT(mesh->nodes()<=X_.size());
  X = X_;
}

template<class TV> void LinearBendingElements<TV>::add_frequency_squared(RawArray<T> frequency_squared) const {
  // We assume edge forces are much stiffer than bending, so our CFL shouldn't matter
}

template<class TV> typename TV::Scalar LinearBendingElements<TV>::strain_rate(RawArray<const TV> V) const {
  // We assume edge forces are much stiffer than bending, so our CFL shouldn't matter
  return 0;
}

template<class TV> static T energy_helper(const SparseMatrix& A,RawArray<const TV> X) {
  GEODE_ASSERT(A.rows()==X.size());
  T diagonal = 0, offdiagonal = 0;
  for (int p=0;p<A.rows();p++) {
    RawArray<const int> J = A.J[p];
    if (J.size())
      diagonal += A.A(p,0)*sqr_magnitude(X[p]);
    for (int a=1;a<J.size();a++)
      offdiagonal += A.A(p,a)*dot(X[p],X[J[a]]);
  }
  return diagonal/2+offdiagonal;
}

template<class TV> typename TV::Scalar LinearBendingElements<TV>::elastic_energy() const {
  return stiffness?stiffness*energy_helper<TV>(*A,X):0;
}

template<class TV> static void add_force_helper(const SparseMatrix& A,const T scale,RawArray<TV> F,RawArray<const TV> X) {
  GEODE_ASSERT(A.rows()<=X.size());
  if (!scale) return;
  for (int p=0;p<A.rows();p++) {
    RawArray<const int> J = A.J[p];
    if (J.size())
      F[p] -= scale*A.A(p,0)*X[p];
    for (int a=1;a<J.size();a++) {
      int q = J[a];
      T entry = scale*A.A(p,a);
      F[p] -= entry*X[q];
      F[q] -= entry*X[p];
    }
  }
}

template<class TV> void LinearBendingElements<TV>::add_elastic_force(RawArray<TV> F) const {
  add_force_helper<TV>(*A,stiffness,F,X);
}

template<class TV> void LinearBendingElements<TV>::add_elastic_differential(RawArray<TV> dF,RawArray<const TV> dX) const {
  add_force_helper<TV>(*A,stiffness,dF,dX);
}

template<class TV> void LinearBendingElements<TV>::add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,d>> dFdX) const {
  GEODE_ASSERT(A->rows()<=dFdX.size());
  if (!stiffness) return;
  for (int p=0;p<A->rows();p++)
    if (A->A.size(p))
      dFdX[p] -= stiffness*A->A(p,0);
}

template<class TV> typename TV::Scalar LinearBendingElements<TV>::damping_energy(RawArray<const TV> V) const {
  return damping?damping*energy_helper<TV>(*A,V):0;
}

template<class TV> void LinearBendingElements<TV>::add_damping_force(RawArray<TV> F,RawArray<const TV> V) const {
  add_force_helper<TV>(*A,damping,F,V);
}

template<class TV> void LinearBendingElements<TV>::structure(SolidMatrixStructure& structure) const {
  for (int p=0;p<A->rows();p++) {
    RawArray<const int> J = A->J[p];
    for (int a=1;a<J.size();a++)
      structure.add_entry(p,J[a]);
  }
}

template<class TV> void add_gradient_helper(const SparseMatrix& A,const T scale,SolidMatrix<TV>& matrix) {
  GEODE_ASSERT(A.rows()<=matrix.size());
  if (!scale) return;
  T minus_scale = -scale;
  for (int p=0;p<A.rows();p++) {
    RawArray<const int> J = A.J[p];
    if (J.size())
      matrix.add_entry(p,minus_scale*A.A(p,0));
    for (int a=1;a<J.size();a++)
      matrix.add_entry(p,J[a],minus_scale*A.A(p,a));
  }
}

template<class TV> void LinearBendingElements<TV>::add_elastic_gradient(SolidMatrix<TV>& matrix) const {
  add_gradient_helper(*A,stiffness,matrix);
}

template<class TV> void LinearBendingElements<TV>::add_damping_gradient(SolidMatrix<TV>& matrix) const {
  add_gradient_helper(*A,damping,matrix);
}

template class LinearBendingElements<Vector<T,2>>;
template class LinearBendingElements<Vector<T,3>>;
}
using namespace geode;

template<int d> static void wrap_helper() {
  typedef Vector<T,d> TV;
  typedef LinearBendingElements<TV> Self;
  Class<Self>(d==2?"LinearBendingElements2d":"LinearBendingElements3d")
    .GEODE_INIT(const typename Self::Mesh&,Array<const TV>)
    .GEODE_FIELD(stiffness)
    .GEODE_FIELD(damping)
    ;
}

void wrap_linear_bending() {
  wrap_helper<2>();
  wrap_helper<3>();
}
