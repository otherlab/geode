#ifdef USE_OPENMESH
#include <other/core/openmesh/smooth.h>
#include <gmm/gmm.h>
#include <other/core/python/wrap.h>
#include <other/core/utility/stream.h>

namespace other {

typedef gmm::row_matrix<gmm::wsvector<real>> Mx;
typedef Vector<real,3> TV;

using std::make_pair;

// split vertices into locked ones (VB) and not locked ones, and compute
// matrix indices for those vertices
static void matrix_permutation(TriMesh const &M,
                               unordered_map<int,VertexHandle,Hasher> &id_to_handle,
                               unordered_map<VertexHandle,int,Hasher> &handle_to_id,
                               Array<VertexHandle> &VB) {
  Array<VertexHandle> V;

  for(auto v : M.vertex_handles()){
    if(M.status(v).locked())
      VB.append(v);
    else
      V.append(v);
  }

  // first all inside ones, append the boundary at the end
  V.append_elements(VB);

  for(int i=0; i < (int)V.size(); ++i){
    id_to_handle.insert(make_pair(i,V[i]));
    handle_to_id.insert(make_pair(V[i],i));
  }

  std::cout << "permute: " << V.size() << " vertices, " << VB.size() << " locked." << std::endl;
}

// make a laplace matrix for the mesh, where all locked vertices come last
static Mx laplace_matrix(TriMesh const &M,
                         unordered_map<VertexHandle,int,Hasher> const &handle_to_id,
                         bool cotan) {

  // compute a laplace matrix for the mesh, using the given index assignment.
  // use cotan weights if cotan is true, otherwise use topological weights.

  int e = handle_to_id.size();

  Mx L(e,e);

  for(auto v : M.vertex_handles()) {
    if (!handle_to_id.count(v))
      continue;

    int i = handle_to_id.find(v)->second;

    auto neighbors = M.vertex_one_ring(v);

    if (neighbors.empty()) {
      L(i,i) = 1.;
      continue;
    }

    real W=0.;

    // compute area
    real A=0.;
    if (cotan) {
      for(auto ohi = M.cvoh_iter(v);ohi;++ohi){
        auto fh = M.face_handle(ohi.handle());

        auto p0 = M.point(v);
        auto p1 = M.point(M.to_vertex_handle(ohi.handle()));
        auto p2 = M.point(M.from_vertex_handle(M.prev_halfedge_handle(ohi.handle())));
        auto d0 = p1-p0;
        auto d1 = p2-p1;
        auto d2 = p0-p2;

        real m0 = d0.sqr_magnitude();
        real m2 = d2.sqr_magnitude();

        d0.normalize(); d1.normalize(); d2.normalize();

        real phi20 = acos(max(-1.,min(1.,dot(d0,-d2))));
        real phi01 = acos(max(-1.,min(1.,dot(-d0,d1))));
        real phi12 = acos(max(-1.,min(1.,dot(d2,-d1))));

        real obtuse = pi*.5;
        if(dot(d0,-d2) >=0 && dot(-d0,d1) >= 0 && dot(-d1,d2) >=0){
          A+=(1/8.) * (m0/tan(phi12) + m2/tan(phi01));
        }else{
          real a = M.area(fh);
          if(phi20 >= obtuse)
            A+=a/2;
          else
            A+=a/4;
        }
      }
    }

    for (auto n : neighbors) {
      OTHER_ASSERT(n.is_valid() && !M.status(n).deleted());
      OTHER_ASSERT(handle_to_id.count(n));
      real w;
      if (cotan)
        w = M.cotan_weight(M.edge_handle(M.halfedge_handle(v,n))); // /(2.*A);
      else
        w = 1.;
      L(i,handle_to_id.find(n)->second) = -w;
      W += w;
    }
    L(i,i) = W;
  }

  return L;
}


Ref<TriMesh> smooth_test(TriMesh &m, bool bilaplace, bool cotan) {
  auto bb = m.bounding_box();
  real zz = bb.sizes().z;

  for (auto v : m.vertex_handles()){
    if (abs(m.point(v).z-bb.center().z)/zz > .4)
      m.status(v).set_locked(true);
  }

  return smooth_mesh(m, bilaplace, cotan);
}




template<class M>
void color_with_eigenvalue(TriMesh &m, M const &A, int nev,
                           unordered_map<VertexHandle,int,Hasher> const &handle_to_id) {

  int n = A.nrows();

  std::vector<real> eig(n);
  gmm::dense_matrix<real> evs(n,n);
  gmm::symmetric_qr_algorithm(A,eig,evs);

  real mx = 0;
  real mn = 1.;
  for (int i =0; i<n; ++i) {
    mx = max(abs(evs(i,nev)),mx);
    mn = min(abs(evs(i,nev)),mn);

    if (i < 20 || n-i < 20)
      std::cout << "ev " << i << ": " << eig[i] << std::endl;
  }

  std::cout << "n = " << n << ", min/max: " << mn << " " << mx << std::endl;

  m.request_vertex_colors();
  for (auto v : m.vertex_handles()) {
    OTHER_ASSERT(handle_to_id.count(v));
    if (!m.status(v).locked()) {
      real e = abs(evs(handle_to_id.find(v)->second,nev));
      real c = (e-mn)/(mx-mn);
      m.set_color(v,to_byte_color(TV(c,0,1-c)));
    } else {
      m.set_color(v,to_byte_color(TV(.5, .5, .5)));
    }
  }
}


Ref<TriMesh> smooth_mesh(TriMesh &m, bool bilaplace, bool cotan) {
  Ref<TriMesh> M = m.copy();
  M->garbage_collection();

  unordered_map<int,VertexHandle,Hasher> id_to_handle;
  unordered_map<VertexHandle,int,Hasher> handle_to_id;
  Array<VertexHandle> VB;

  matrix_permutation(M, id_to_handle, handle_to_id, VB);

  int e = handle_to_id.size();
  int n = e - VB.size();

  std::cout << "e = " << e << ", n = " << n << std::endl;

  Mx L = laplace_matrix(M, handle_to_id, cotan);

  // make a system matrix
  Mx U(e,e);
  if (bilaplace)
    gmm::mult(gmm::transposed(L),L,U);
  else
    gmm::mult(gmm::identity_matrix(),L,U);

  // results
  std::vector<TV> output(n);

  // chop up matrix
  auto A = gmm::sub_matrix(U, gmm::sub_interval(0, n));
  auto B = gmm::sub_matrix(U, gmm::sub_interval(0, n), gmm::sub_interval(n, e-n));

  //color_with_eigenvalue(M, A, 0, handle_to_id);

  // solve for x, y, z
  for (int i =0; i < 3; ++i) {

    std::vector<double> X(n);
    gmm::iteration iter(1e-9);
    iter.set_maxiter(10000);

    vector<double> y(e-n);
    vector<double> RHS(n);

    for(auto v : VB)
      y[handle_to_id.find(v)->second - n] = M->point(v)[i];

    gmm::mult(B,y,RHS);
    gmm::scale(RHS,-1.);

    gmm::cg(A,X,RHS,gmm::identity_matrix(),gmm::identity_matrix(),iter);

    for(int j=0;j<(int)X.size();++j)
      output[j][i] = X[j];
  }

  // write result back to M
  for(int i=0;i<(int)output.size();++i) {
    OTHER_ASSERT(id_to_handle.count(i));
    auto vh = id_to_handle.find(i)->second;
    OTHER_ASSERT(vh.is_valid() && !M->status(vh).locked());
    M->point(vh) = output[i];
  }

  return M;
}

}

using namespace other;

void wrap_smooth() {
  OTHER_FUNCTION(smooth_mesh)
  OTHER_FUNCTION(smooth_test)
}

#endif
