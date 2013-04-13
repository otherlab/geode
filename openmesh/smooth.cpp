#ifdef USE_OPENMESH
#include <other/core/openmesh/smooth.h>
#include <gmm/gmm.h>
#include <other/core/python/wrap.h>
#include <other/core/utility/stream.h>

namespace other {

typedef gmm::row_matrix<gmm::wsvector<real>> Mx;
typedef Vector<real,3> TV;

using std::make_pair;

#define VORONOI_AREA 1

/ split vertices into locked ones (VB) and not locked ones, and compute
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

#ifdef VORONOI_AREA
static real mixed_voronoi_area(TriMesh const & m, const VertexHandle vh){
  real A = 0.;
  for(auto ohi = m.cvoh_iter(vh);ohi;++ohi){
    auto fh = m.face_handle(ohi.handle());

    auto p0 = m.point(vh);
    auto p1 = m.point(m.to_vertex_handle(ohi.handle()));
    auto p2 = m.point(m.from_vertex_handle(m.prev_halfedge_handle(ohi.handle())));
    auto d0 = p1-p0;
    auto d1 = p2-p1;
    auto d2 = p0-p2;

    real m0 = d0.sqr_magnitude();
    real m2 = d2.sqr_magnitude();

    d0.normalize(); d1.normalize(); d2.normalize();

    real phi20 = acos( max(-1.,min(1.,dot(d0,-d2))) );
    real phi01 = acos( max(-1.,min(1.,dot(-d0,d1))) );
    real phi12 = acos(max(-1.,min(1.,dot(d2,-d1))));

    real obtuse = pi*.5;
    if(dot(d0,-d2) >=0 && dot(-d0,d1) >= 0 && dot(-d1,d2) >=0){
      A+=(1/8.) * (m0/tan(phi12) + m2/tan(phi01));
    }else{
      real a = m.area(fh);
      if(phi20 >= obtuse)
        A+=a/2.;
      else A+=a/4.;
    }
  }
  return A;
}
#endif


Ref<TriMesh> smooth_mesh(TriMesh const &m, real dt, int iters, bool bilaplace) {
  Ref<TriMesh> M = m.copy();
  M->garbage_collection();

  auto bb = M->bounding_box();
  real zz = bb.sizes().z;

  for (auto v : M->vertex_handles()){
    if(abs(M->point(v).z-bb.center().z) > .3*zz) M->status(v).set_locked(true);
  }

  Array<VertexHandle> VI;
  Array<VertexHandle> VB;

  unordered_map<int,VertexHandle,Hasher> id_to_handle;
  unordered_map<VertexHandle,int,Hasher> handle_to_id;

  for(auto v : M->vertex_handles()){
    if(M->status(v).locked()) VB.append(v);
    else VI.append(v);
  }

  Array<VertexHandle> V(VI);
  V.append_elements(VB);

  for(int i=0; i < (int)V.size(); ++i){
    id_to_handle.insert(make_pair(i,V[i]));
    handle_to_id.insert(make_pair(V[i],i));
  }

  int n = VI.size();
  int e = V.size();

  //calculate original cotangent weights; these will not change with timesteps
  Mx L(e,e);
  for(auto v : M->vertex_handles()){
    auto neighbors = M->vertex_one_ring(v);
    OTHER_ASSERT(neighbors.size() && v.is_valid());
    int i = handle_to_id[v];
    real W=0.;
    for(auto n : neighbors){
      OTHER_ASSERT(n.is_valid() && !M->status(n).deleted());

      auto heh = M->halfedge_handle(v,n);
      real w = max(0.,M->cotan_weight(M->edge_handle(heh)));
      L(i,handle_to_id[n]) = -w;
      W+=w;
    }
    L(i,i) = W;
  }

  gmm::identity_matrix PS;
  gmm::identity_matrix PR;

  std::vector<TV> Y(e-n);
  for(auto v : VB)
    Y[handle_to_id[v] - n] = M->point(v); //offset by n since we only grab boundaries


  for(int j=0; j < max(1,iters); ++j){
    // recalculate mixed voronoi region weights into a diagonal matrix
    Mx D(e,e);
    for(auto v : M->vertex_handles()){
      auto neighbors = M->vertex_one_ring(v);
      OTHER_ASSERT(neighbors.size() && v.is_valid());

      int i = handle_to_id[v];
      #ifdef VORONOI_AREA
      real va = mixed_voronoi_area(*M,v);
      OTHER_ASSERT(va > 1e-8);
      D(i,i) = 1./(4*va);
      #else
      real W=0.;
      for(auto n : neighbors){
        OTHER_ASSERT(n.is_valid() && !M->status(n).deleted());

        auto heh = M->halfedge_handle(v,n);
        real w = .5 * (M->area(M->face_handle(heh)) + M->area(M->face_handle(M->opposite_halfedge_handle(heh))));
        D(i,handle_to_id[n]) = w;
        W+=w;
      }
      D(i,i) = 1./W;
      #endif
    }

    Mx U(e,e);
    Mx K(e,e);
    gmm::mult(D,L,U);
    if(bilaplace) gmm::mult(U,U,K);
    else gmm::mult(gmm::identity_matrix(),U,K);

    gmm::scale(K,-1.);
    auto A_ = gmm::sub_matrix(K, gmm::sub_interval(0, n));
    auto B_ = gmm::sub_matrix(K, gmm::sub_interval(0, n),gmm::sub_interval(n, e-n));

    Mx A(n,n);
    gmm::copy(A_,A);

    Mx B(n,e-n);
    gmm::copy(B_,B);

    // -dt*A
    gmm::scale(A,-dt);
    // I-dt*A
    gmm::add(gmm::identity_matrix(),A,A);

    // dt*B
    gmm::scale(B,dt);

    // accumulate X^n
    std::vector<TV> Xn(n);
    for(auto v : VI)
      Xn[handle_to_id[v]] = M->point(v);

    std::vector<TV> output(n);

    for(int i =0; i < 3; ++i){
      // storage for result
      std::vector<double> Xi(n);

      // i-components of X^n
      std::vector<double> Xni(n);
      for(int idx =0; idx < (int)Xn.size(); ++idx)
        Xni[idx]=Xn[idx][i];

      // i-components of Y
      std::vector<double> Yi(e-n);
      for(int idx = 0; idx < (int)Y.size(); ++idx)
        Yi[idx] = Y[idx][i]; //offset by n

      // dt*B*y
      std::vector<double> BYi(n);
      gmm::mult(B,Yi,BYi);

      // X^n + dt*B*y
      std::vector<double> RHS(n);
      gmm::add(Xni,BYi,RHS);

      gmm::iteration it(10E-6);

      gmm::cg(A,Xi,RHS,PS,PR,it);

      for(int j=0;j<(int)Xi.size();++j)
        output[j][i] = Xi[j];
    }

    for(int i=0;i<(int)output.size();++i){
      auto vh = id_to_handle[i];
      OTHER_ASSERT(!M->status(vh).locked());
      M->point(vh) = output[i];
    }
  }

  return M;
}

}

using namespace other;

void wrap_smooth() {
  OTHER_FUNCTION(smooth_mesh)
}

#endif
