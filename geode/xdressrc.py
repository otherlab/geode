# Xdress definitions for geode

from xdress.types.system import TypeSystem
from xdress.utils import apiname

# Geode!
package    = 'geode'
packagedir = 'geode'

# Start off with an empty type system
ts = TypeSystem.empty()

# Teach xdress how to use from_python
'''
def geode_convert(name,args=None):
ts.register_class("Array", ('T',), cython_py2c=(
  '{proxy_name}.from_python({var})',
  '{proxy_name}'
  ))
'''
ts.register_class('Vector',('T','d'),
  cpp_type='Vector',
  cython_cy_type='aaaaaa',cython_py_type='bbbbbbbbb',cython_c_type='ccccccc',
  cython_cimport=(None,),
  cython_pyimport=(None,),
  cython_cyimport=(None,),
  cython_py2c=('FromPython[{t.type}].convert({var})',False),
  cython_c2py=('to_python({var})',))

def concat(*args):
  return [b for a in args for b in a]

def names(header,names):
  src = ['geode/'+header]
  return [apiname(n,srcfiles=src,language='c++',tarbase='wrap',sidecars=()) for n in names]

# Wrapped functions

functions = concat(
  names('exact/predicates.h',('predicate_tests',)),
  names('utility/Log.h','''log_initialized log_configure log_cache_initial_output log_copy_to_file log_finish
                           log_push_scope log_pop_scope log_print log_error log_flush'''.split()),
  names('utility/repr.h',('str_repr_test',)),
  names('utility/openmp.h','partition_loop_test large_partition_loop_test'.split()),
  names('utility/format.h',('format_test',)),
  names('utility/process.h','cpu_times memory_usage max_memory_usage'.split()),
)

'''
# utility/resource.h
void wrap_resource() {
#ifdef GEODE_PYTHON
  // Python is active, so set the executable path to the script directory
  Ref<> sys = steal_ref_check(PyImport_ImportModule("sys"));
  Ref<> argv = python_field(sys,"argv");
  Ref<> argv0 = steal_ref_check(PySequence_GetItem(&*argv,0));
  helper() = path::dirname(from_python<string>(argv0));
#endif
  GEODE_FUNCTION(resource_path)
  GEODE_FUNCTION_2(resource_py,static_cast<string(*)(const string&)>(resource))
}

# utility/base64
void wrap_base64() {
  GEODE_FUNCTION_2(base64_encode,static_cast<string(*)(const string&)>(base64_encode))
  GEODE_FUNCTION(base64_decode)
}
'''

# uint128.h
#GEODE_FUNCTION(uint128_test)
#GEODE_FUNCTION_2(uint128_str_test,static_cast<string(*)(uint128_t)>(str))

'''
# constructions.h
GEODE_FUNCTION(construction_tests)

# Exact.h
GEODE_FUNCTION(fast_exact_tests)

# circle_csg.h
void wrap_circle_csg() {
  GEODE_FUNCTION(split_circle_arcs)
  GEODE_FUNCTION(exact_split_circle_arcs)
  GEODE_FUNCTION(canonicalize_circle_arcs)
  GEODE_FUNCTION_2(circle_arc_area,static_cast<real(*)(Nested<const CircleArc>)>(circle_arc_area))
  GEODE_FUNCTION(preprune_small_circle_arcs)
  reverse_arcs
#ifdef GEODE_PYTHON
  GEODE_FUNCTION(_set_circle_arc_dtypes)
  GEODE_FUNCTION(circle_arc_quantize_test)
  GEODE_FUNCTION(random_circle_quantize_test)
  GEODE_FUNCTION(single_circle_handling_test)
#endif
}

# vector/python.cpp
void wrap_vector() {
#ifdef GEODE_PYTHON
  GEODE_FUNCTION_2(min_magnitude,min_magnitude_python)
  GEODE_FUNCTION_2(max_magnitude,max_magnitude_python)

  // for testing purposes
  GEODE_FUNCTION(vector_test)
  GEODE_FUNCTION(matrix_test)
  GEODE_FUNCTION(vector_stream_test)
#endif
}

# array/Array.cpp
void wrap_array() {
  // for testing purposes
  GEODE_FUNCTION(empty_array_test)
  GEODE_FUNCTION(array_test)
  GEODE_FUNCTION(nested_test)
  GEODE_FUNCTION(nested_convert_test)
  GEODE_FUNCTION(const_array_test)
}

# math/integer_log.h
GEODE_FUNCTION_2(integer_log,static_cast<int(*)(uint64_t)>(integer_log))
GEODE_FUNCTION_2(popcount,static_cast<int(*)(uint64_t)>(popcount))
GEODE_FUNCTION_2(min_bit,static_cast<uint64_t(*)(uint64_t)>(min_bit))

# exact/perturb.h
GEODE_FUNCTION(perturb_monomials)
GEODE_FUNCTION_2(perturbed_sign_test_1,perturbed_sign_test<1>)
GEODE_FUNCTION_2(perturbed_sign_test_2,perturbed_sign_test<2>)
GEODE_FUNCTION_2(perturbed_sign_test_3,perturbed_sign_test<3>)
GEODE_FUNCTION(in_place_interpolating_polynomial_test)
GEODE_FUNCTION(snap_divs_test)
GEODE_FUNCTION(perturbed_ratio_test)

# exact/Interval.h
interval_tests

# exact/polygon_csg.h
GEODE_FUNCTION(split_polygons)

# geometry/surface_levelset.h
GEODE_FUNCTION_2(evaluate_surface_levelset,evaluate_surface_levelset_python)
GEODE_FUNCTION(slow_evaluate_surface_levelset)

# exact/circle_offsets.h
GEODE_FUNCTION(offset_arcs)
GEODE_FUNCTION(offset_shells)

# exact/delaunay.h
GEODE_FUNCTION_2(delaunay_points_py,delaunay_points)
GEODE_FUNCTION(greedy_nonintersecting_edges)
GEODE_FUNCTION(chew_fan_count)

# geometry/Segment.h
void wrap_segment() {
  GEODE_FUNCTION_2(segment_tests_2d,segment_tests<2>)
  GEODE_FUNCTION_2(segment_tests_3d,segment_tests<3>)
}

# geometry/Implicit.h
template<> GEODE_DEFINE_TYPE(Implicit<Vector<T,1> >)
template<> GEODE_DEFINE_TYPE(Implicit<Vector<T,2> >)
template<> GEODE_DEFINE_TYPE(Implicit<Vector<T,3> >)
template<int d> static void wrap_helper() {
  typedef Vector<T,d> TV;
  typedef Implicit<TV> Self;

  Class<Self>("Implicit")
    .GEODE_FIELD(d)
    .GEODE_METHOD(phi)
    .GEODE_METHOD(normal)
    .GEODE_METHOD(lazy_inside)
    .GEODE_METHOD(surface)
    .GEODE_METHOD(bounding_box)
    .GEODE_REPR()
    ;
}

void wrap_implicit() {
  wrap_helper<1>();
  wrap_helper<2>();
  wrap_helper<3>();
}

# geometry/FrameImplicit.h
template<> GEODE_DEFINE_TYPE(FrameImplicit<Vector<T,2> >)
template<> GEODE_DEFINE_TYPE(FrameImplicit<Vector<T,3> >)
template<int d> static void wrap_helper() {
  typedef Vector<T,d> TV;
  typedef FrameImplicit<TV> Self;
  static const string name = format("FrameImplicit%dd",d);

  Class<Self>(name.c_str())
    .GEODE_INIT(Frame<TV>,const Implicit<TV>&)
    ;
}
  
void wrap_frame_implicit() {
  wrap_helper<2>();
  wrap_helper<3>();

# geometry/platonic.h
GEODE_FUNCTION(icosahedron_mesh)
GEODE_FUNCTION_2(sphere_mesh_py,sphere_mesh)
GEODE_FUNCTION(double_torus_mesh)

# geometry/Bezier
void wrap_bezier() {
  {   
    typedef Knot<2> Self;
    Class<Self>("Knot")
      .GEODE_INIT()
      .GEODE_FIELD(pt)
      .GEODE_FIELD(tangent_in)
      .GEODE_FIELD(tangent_out)
      ;
  }   
  {
    typedef Bezier<2> Self;
    typedef Array<Vector<real,2>>(Self::*eval_t)(int)const;
    Class<Self>("Bezier")
      .GEODE_INIT()
      .GEODE_FIELD(knots)
      .GEODE_METHOD(t_max)
      .GEODE_METHOD(t_min)
      .GEODE_METHOD(closed)
      .GEODE_METHOD(close)
      .GEODE_METHOD(fuse_ends)
      .GEODE_OVERLOADED_METHOD(eval_t, evaluate)
      .GEODE_METHOD(append_knot)
      ;
  }
}

# geometry/polygon.h
GEODE_FUNCTION_2(polygon_area,polygon_area_py)
GEODE_FUNCTION(polygons_from_index_list)
GEODE_FUNCTION(canonicalize_polygons)

# geometry/BoxTree.h
void wrap_box_tree() {
  {typedef Vector<real,2> TV;
  typedef BoxTree<TV> Self;
  Class<Self>("BoxTree2d")
    .GEODE_INIT(RawArray<const TV>,int)
    .GEODE_FIELD(p)
    .GEODE_METHOD(check)
    ;}

  {typedef Vector<real,3> TV;
  typedef BoxTree<TV> Self;
  Class<Self>("BoxTree3d")
    .GEODE_INIT(RawArray<const TV>,int)
    .GEODE_FIELD(p)
    .GEODE_METHOD(check)
    ;}
}

# geometry/ParticleTree.h
template<int d> static void wrap_helper() {
  typedef Vector<T,d> TV;
  typedef ParticleTree<TV> Self;
  Class<Self>(d==2?"ParticleTree2d":"ParticleTree3d")
    .GEODE_INIT(Array<const TV>,int)
    .GEODE_FIELD(X)
    .GEODE_METHOD(update)
    .GEODE_METHOD(remove_duplicates)
    .GEODE_METHOD_2("closest_point",closest_point_py)
    ;
}

void wrap_particle_tree() {
  wrap_helper<2>();
  wrap_helper<3>();
}

# geometry/AnalyticImplicit.h
template<int d> static void wrap_helper() {
  typedef Vector<T,d> TV;

  {typedef AnalyticImplicit<Sphere<TV> > Self;
  Class<Self>(d==2?"Sphere2d":"Sphere3d")
    .GEODE_INIT(TV,T)
    .GEODE_METHOD(volume)
    ;}

  {typedef AnalyticImplicit<Capsule<TV> > Self;
  Class<Self>(d==2?"Capsule2d":"Capsule3d")
    .GEODE_INIT(TV,TV,T)
    .GEODE_METHOD(volume)
    ;}
}
template<int d> static void wrap_box_helper() {
  typedef Vector<T,d> TV;
  typedef AnalyticImplicit<Box<TV> > Self;
  Class<Self>(d==1?"Box1d":d==2?"Box2d":"Box3d")
    .GEODE_INIT(TV,TV)
    .GEODE_FIELD(min)
    .GEODE_FIELD(max)
    .GEODE_METHOD(sizes)
    .GEODE_METHOD(clamp)
    .GEODE_METHOD(center)
    .GEODE_METHOD(volume)
    .template method<Box<TV>(Box<TV>::*)(T)const>("thickened",&Self::thickened)
    .template method<void(Box<TV>::*)(const TV&)>("enlarge",&Self::enlarge)
    ;

  geode::python::function(d==1?"empty_box_1d":d==2?"empty_box_2d":"empty_box_3d",Box<TV>::empty_box);
}
void wrap_analytic_implicit() {
  wrap_helper<2>();
  wrap_helper<3>();
  wrap_box_helper<1>();
  wrap_box_helper<2>();
  wrap_box_helper<3>();

  typedef Vector<T,3> TV;

  {typedef AnalyticImplicit<Plane<T>> Self;
  Class<Self>("Plane")
    .GEODE_INIT(TV,TV)
    ;}

  {typedef AnalyticImplicit<Cylinder> Self;
  Class<Self>("Cylinder")
    .GEODE_INIT(TV,TV,T)
    ;}
}

# geometry/SimplexTree.h
template<class TV,int d> static void wrap_helper() {
  typedef SimplexTree<TV,d> Self;
  static const string name = format("%sTree%dd",(d==1?"Segment":"Triangle"),TV::m);
  Class<Self>(name.c_str())
    .GEODE_INIT(const typename Self::Mesh&,Array<const TV>,int)
    .GEODE_FIELD(mesh)
    .GEODE_FIELD(X)
    .GEODE_METHOD(update)
    .GEODE_METHOD(closest_point)
    ;
}

void wrap_simplex_tree() {
  wrap_helper<Vector<real,2>,1>();
  wrap_helper<Vector<real,2>,2>();
  wrap_helper<Vector<real,3>,1>();
  wrap_helper<Vector<real,3>,2>();
  GEODE_FUNCTION_2(ray_traversal_test,ray_traversal_test<real,3>)
}

# geometry/ThickShell.h
void wrap_thick_shell() {
  typedef ThickShell Self;
  Class<Self>("ThickShell")
    .GEODE_INIT(Ref<>,Array<const TV>,Array<const T>)
    ;
}

# vector/Matrix.h
void wrap_matrix() {
#ifdef GEODE_PYTHON
  using namespace python; 
  function("_set_matrix_type",set_matrix_type);
  GEODE_FUNCTION_2(fast_singular_values,fast_singular_values_py)
#endif  
}

# vector/Frame.h
void wrap_frame() {
#ifdef GEODE_PYTHON
  using namespace python;
  function("_set_frame_type",set_frame_type);
  function("frame_test_2d",frame_test<Vector<real,2>>);
  function("frame_test_3d",frame_test<Vector<real,3>>);
  function("frame_array_test_2d",frame_array_test<Vector<real,2>>);
  function("frame_array_test_3d",frame_array_test<Vector<real,3>>);
  function("frame_interpolation_2d",frame_interpolation<Vector<real,2>>);
  function("frame_interpolation_3d",frame_interpolation<Vector<real,3>>);
#endif
}

# vector/Frame.h
void wrap_rotation() {
#ifdef GEODE_PYTHON
  using namespace python;
  function("_set_rotation_types",set_rotation_types);
  function("rotation_test_2d",rotation_test<Vector<real,2>>);
  function("rotation_test_3d",rotation_test<Vector<real,3>>);
  function("rotation_array_test_2d",rotation_array_test<Vector<real,2>>);
  function("rotation_array_test_3d",rotation_array_test<Vector<real,3>>);
  GEODE_FUNCTION(rotation_from_matrix);
  GEODE_FUNCTION(rotation_from_euler_angles_3d);
  GEODE_FUNCTION(rotation_euler_angles_3d);
#endif
}

# vector/Register.h
void wrap_register() {
#ifdef GEODE_PYTHON
  GEODE_FUNCTION_2(rigid_register,rigid_register_python)
  GEODE_FUNCTION_2(affine_register,affine_register_python)
#endif
}

# vector/SparseMatrix.h
void wrap_sparse_matrix() {
    typedef SparseMatrix Self;
    Class<Self>("SparseMatrix")
        .GEODE_INIT(Nested<int>,Array<T>)
        .GEODE_METHOD(rows)
        .GEODE_METHOD(columns)
        .GEODE_FIELD(J)
        .GEODE_FIELD(A)
        .GEODE_METHOD_2("multiply",multiply_python)
        .GEODE_METHOD(solve_forward_substitution)
        .GEODE_METHOD(solve_backward_substitution)
        .GEODE_METHOD(incomplete_cholesky_factorization)
        .GEODE_METHOD(gauss_seidel_solve)
        ;
}

# vector/SolidMatrix.h
template<int d> static void wrap_helper() {
  {typedef SolidMatrixBase<Vector<T,d>> Self;
  Class<Self>(d==2?"SolidMatrixBase2d":"SolidMatrixBase3d")
    .GEODE_METHOD(multiply)
    ;}

  {typedef SolidMatrix<Vector<T,d>> Self;
  Class<Self>(d==2?"SolidMatrix2d":"SolidMatrix3d")
    .GEODE_INIT(const SolidMatrixStructure&)
    .GEODE_METHOD(copy)
    .GEODE_METHOD(size)
    .GEODE_METHOD(zero)
    .GEODE_METHOD(scale)
    .method("add_entry",static_cast<void(Self::*)(int,int,const Matrix<T,d>&)>(&Self::add_entry))
    .GEODE_METHOD(add_scalar)
    .GEODE_METHOD(add_diagonal_scalars)
    .GEODE_METHOD(add_partial_scalar)
    .GEODE_METHOD(add_outer)
    .GEODE_METHOD(entries)
    .GEODE_METHOD(inverse_block_diagonal)
    .GEODE_METHOD(inner_product)
    .GEODE_METHOD(diagonal_range)
    .GEODE_METHOD(dense)
    ;}

  {typedef SolidDiagonalMatrix<Vector<T,d>> Self;
  Class<Self>(d==2?"SolidDiagonalMatrix2d":"SolidDiagonalMatrix3d")
    .GEODE_METHOD(inner_product)
    ;}
}
void wrap_solid_matrix() {
  {typedef SolidMatrixStructure Self;
  Class<Self>("SolidMatrixStructure")
    .GEODE_INIT(int)
    .GEODE_METHOD(copy)
    .GEODE_METHOD(add_entry)
    .GEODE_METHOD(add_outer)
    ;}
  wrap_helper<2>();
  wrap_helper<3>();
}

# array/stencil.h
void wrap_stencil() {
  typedef uint8_t T;
  typedef const boost::function<T (const Array<T,2>, Vector<int,2> const &)> ftype;
  typedef void(*stencil_ftype)(ftype &, int, const Array<T,2>);
  GEODE_FUNCTION_2(apply_stencil_uint8, static_cast<stencil_ftype>(apply_stencil<ftype, T, 2>));
  typedef MaxStencil<T> Self;
  Class<Self>("MaxStencil_uint8")
    .GEODE_INIT(int)
    .GEODE_FIELD(r)
    .GEODE_CALL(const Array<const typename Self::value_type,2>, Vector<int,2> const &)
    ;
}

# image/color_utils.h
void wrap_color_utils() {
  GEODE_FUNCTION_2(wheel_color,wheel_color_py) // NdArray
}

# value/PropManager.h
void wrap_prop_manager() {
#ifdef GEODE_PYTHON
  typedef PropManager Self;
  Class<Self>("PropManager")
    .GEODE_INIT()
    .method("add_existing",static_cast<PropBase&(Self::*)(PropBase&)>(&Self::add))
    .method("get",static_cast<PropBase&(Self::*)(const string&)const>(&Self::get))
    .GEODE_METHOD_2("add",add_python)
    .GEODE_METHOD_2("get_or_add",get_or_add_python)
    .GEODE_METHOD(contains)
    .GEODE_CONST_FIELD(items)
    .GEODE_CONST_FIELD(order)
    .GEODE_FIELD(frozen)
    .getattr()
    ;
#endif
}

# image/Image.h
void wrap_Image()
{
  using namespace geode;
  using namespace python;
  typedef real T;

  typedef Image<T> Self;
  Class<Self>("Image")
    .GEODE_METHOD(read)
    .GEODE_METHOD(write)
    .GEODE_METHOD(gamma_compress)
    .GEODE_METHOD(dither)
    .GEODE_METHOD(median)
    .GEODE_METHOD(is_supported)
    ;
}

# image/MovFile.h
void wrap_mov()
{
    typedef MovWriter Self;
    Class<Self>("MovWriter")
        .GEODE_INIT(const string&,int)
        .GEODE_METHOD(add_frame)
        .GEODE_METHOD(write_footer)
        .GEODE_METHOD(enabled)
        ;
}

# value/Value.h
  typedef ValueBase Self;
  Class<Self>("Value")
    .GEODE_CALL()
    .GEODE_GET(name)
    .GEODE_METHOD(dirty)
    .GEODE_METHOD(dump)
    .GEODE_METHOD(dependents)
    .GEODE_METHOD(all_dependents)
    .GEODE_METHOD(dependencies)
    .GEODE_METHOD(all_dependencies)
    .GEODE_METHOD(signal)
    .GEODE_METHOD(is_prop)
    // The following work only if the object is a Prop
    .GEODE_METHOD(set)
    .property("help",&Self::get_help)
    .property("hidden",&Self::get_hidden)
    .property("required",&Self::get_required)
    .property("category",&Self::get_category)
    .property("abbrev",&Self::get_abbrev)
    .property("allowed",&Self::get_allowed)
    .property("default",&Self::get_default)
    .GEODE_METHOD(set_help)
    .GEODE_METHOD(set_hidden)
    .GEODE_METHOD(set_required)
    .GEODE_METHOD(set_abbrev)
    .GEODE_METHOD(set_allowed)
    .GEODE_METHOD(set_category)
    .GEODE_METHOD_2("set_min",set_min_py)
    .GEODE_METHOD_2("set_max",set_max_py)
    .GEODE_METHOD_2("set_step",set_step_py)
    ;

  GEODE_FUNCTION(value_test)
  GEODE_FUNCTION(value_ptr_test)
}

# value/Prop.h
void wrap_prop() {
#ifdef GEODE_PYTHON 
  GEODE_FUNCTION(make_prop)
  GEODE_FUNCTION(make_prop_shape)
  GEODE_FUNCTION(unusable_prop_test)
#endif
}

# force/Springs.h
void wrap_springs() {
  typedef real T;
  typedef Vector<T,3> TV;
  typedef Springs<TV> Self;
  Class<Self>("Springs")
    .GEODE_INIT(Array<const Vector<int,2>>,Array<const T>,Array<const TV>,NdArray<const T>,NdArray<const T>)
    .GEODE_METHOD(restlengths)
    .GEODE_FIELD(springs)
    .GEODE_FIELD(resist_compression)
    .GEODE_FIELD(strain_range)
    .GEODE_FIELD(off_axis_damping)
    .GEODE_METHOD(limit_strain)
    ;
}

# force/SurfacePins.h
void wrap_surface_pins() {
  typedef SurfacePins Self;
  Class<Self>("SurfacePins")
    .GEODE_INIT(Array<const int>,Array<const T>,TriangleSoup&,Array<const TV>,NdArray<const T>,NdArray<const T>)
    .GEODE_METHOD(closest_points)
    ;
}

# force/Force.h
template<int d> static void wrap_helper() {
  typedef Force<Vector<T,d>> Self;
  Class<Self>(d==2?"Force2d":"Force3d")
    .GEODE_FIELD(d)
    .GEODE_METHOD(nodes)
    .GEODE_METHOD(update_position)
    .GEODE_METHOD(elastic_energy)
    .GEODE_METHOD(add_elastic_force)
    .GEODE_METHOD(add_elastic_differential)
    .GEODE_METHOD(damping_energy)
    .GEODE_METHOD(add_damping_force)
    .GEODE_METHOD(add_frequency_squared)
    .GEODE_METHOD(strain_rate)
    .GEODE_METHOD(structure)
    .GEODE_METHOD(add_elastic_gradient)
    .GEODE_METHOD(add_damping_gradient)
    .GEODE_METHOD(elastic_gradient_block_diagonal_times)
    ;
} 
void wrap_Force() {
  wrap_helper<2>();
  wrap_helper<3>();
}

# force/AirPressure
void wrap_air_pressure() {
  typedef AirPressure Self;
  Class<Self>("AirPressure")
    .GEODE_INIT(Ref<TriangleSoup>,Array<const TV>,bool,int)
    .GEODE_FIELD(temperature)
    .GEODE_FIELD(amount)
    .GEODE_FIELD(pressure)
    .GEODE_FIELD(skip_rotation_terms)
    .GEODE_FIELD(initial_volume)
    ;
}

# force/SimpleShell.h
void wrap_simple_shell() {
  typedef SimpleShell Self;
  Class<Self>("SimpleShell")
    .GEODE_INIT(const TriangleSoup&,RawArray<const Matrix<T,2>>,T)
    .GEODE_FIELD(density)
    .GEODE_FIELD(stretch_stiffness)
    .GEODE_FIELD(shear_stiffness)
    .GEODE_FIELD(F_threshold)
    ;
}

# force/AxisPins.h
void wrap_axis_pins() {
  typedef AxisPins Self;
  Class<Self>("AxisPins")
    .GEODE_INIT(Array<const int>,Array<const T>,Array<const TV>,NdArray<const T>,NdArray<const T>)
    .GEODE_FIELD(axis)
    ;
}

# force/Gravity.h
void wrap_gravity() {
  typedef real T;
  typedef Vector<T,3> TV;
  typedef Gravity<TV> Self;
  Class<Self>("Gravity")
    .GEODE_INIT(Array<const T>)
    .GEODE_FIELD(gravity)
    ;
}

# force/Pins
void wrap_pins() {
  typedef Pins Self;
  Class<Self>("Pins")
    .GEODE_INIT(Array<const int>,Array<const T>,Array<const TV>,NdArray<const T>,NdArray<const T>)
    ;
}

# force/NeoHookean.h
oid wrap_neo_hookean() {
  {typedef NeoHookean<T,2> Self;
  Class<Self>("NeoHookean2d")
    .GEODE_INIT(T,T,T,T)
    ;}

  {typedef NeoHookean<T,3> Self;
  Class<Self>("NeoHookean3d")
    .GEODE_INIT(T,T,T,T)
    ;}
}

# force/LinearFiniteVolume
void wrap_linear_finite_volume() {
  {typedef LinearFiniteVolume<Vector<T,2>,2> Self;
  Class<Self>("LinearFiniteVolume2d")
    .GEODE_INIT(Array<const Vector<int,3>>,Array<const Vector<T,2>>,T,T,T,T)
    ;}

  {typedef LinearFiniteVolume<Vector<T,3>,2> Self;
  Class<Self>("LinearFiniteVolumeS3d")
    .GEODE_INIT(Array<const Vector<int,3>>,Array<const Vector<T,3>>,T,T,T,T)
    ;}

  {typedef LinearFiniteVolume<Vector<T,3>,3> Self;
  Class<Self>("LinearFiniteVolume3d")
    .GEODE_INIT(Array<const Vector<int,4>>,Array<const Vector<T,3>>,T,T,T,T)
    ;}
}

# force/BindingSprings.h
void wrap_binding_springs() {
  {typedef BindingSprings<2> Self;
  Class<Self>("BindingSprings2d")
    .GEODE_INIT(Array<const int>,Array<const Vector<int,2>>,Array<const Vector<T,2>>,Array<const T>,NdArray<const T>,Nd
    ;}

  {typedef BindingSprings<3> Self;
  Class<Self>("BindingSprings3d")
    .GEODE_INIT(Array<const int>,Array<const Vector<int,3>>,Array<const Vector<T,3>>,Array<const T>,NdArray<const T>,Nd
    ;}
}

# force/CubicHinges
template<int d> static void wrap_helper() {
  typedef Vector<T,d+1> TV;
  typedef CubicHinges<TV> Self;
  Class<Self>(d==1?"CubicHinges2d":"CubicHinges3d")
    .GEODE_INIT(Array<const Vector<int,d+2>>,RawArray<const T>,RawArray<const TV>)
    .GEODE_METHOD(slow_elastic_energy)
    .GEODE_METHOD(angles)
    .GEODE_FIELD(stiffness)
    .GEODE_FIELD(damping)
    .GEODE_FIELD(simple_hessian)
    ;
}

void wrap_cubic_hinges() {
  wrap_helper<1>();
  wrap_helper<2>();
}

# force/LinearBendingElements.h
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

# force/PlasticityModel.h
void wrap_plasticity_model() {
  {typedef PlasticityModel<T,2> Self;
  Class<Self>("PlasticityModel2d");}

  {typedef PlasticityModel<T,3> Self;
  Class<Self>("PlasticityModel3d");}
}

# force/ParticleBindingSprings.h
void wrap_particle_binding_springs() {
  typedef ParticleBindingSprings Self;
  Class<Self>("ParticleBindingSprings")
    .GEODE_INIT(Array<const Vector<int,2>>,Array<const T>,NdArray<const T>,NdArray<const T>)
    ;
}

# force/StrainMeasureHex
void wrap_strain_measure_hex() {
  typedef StrainMeasureHex Self;
  Class<Self>("StrainMeasureHex")
    .GEODE_INIT(Array<const Vector<int,8>>,Array<const Vector<T,3>>)
    .GEODE_FIELD(elements)
    ; 
}

# force/LinearFiniteVolumeHex.h
void wrap_linear_finite_volume_hex() {
  typedef LinearFiniteVolumeHex Self;
  Class<Self>("LinearFiniteVolumeHex")
    .GEODE_INIT(const StrainMeasureHex&,T,T,T,T)
    ;
}

# force/StrainMeasure
void wrap_strain_measure() {
  {typedef StrainMeasure<T,2> Self;
  Class<Self>("StrainMeasure2d")
    .GEODE_INIT(Array<const Vector<int,3>>,RawArray<const T,2>)
    .GEODE_FIELD(elements)
    .GEODE_METHOD(print_altitude_statistics)
    ;}

  {typedef StrainMeasure<T,3> Self;
  Class<Self>("StrainMeasure3d")
    .GEODE_INIT(Array<const Vector<int,4>>,RawArray<const T,2>)
    .GEODE_FIELD(elements)
    .GEODE_METHOD(print_altitude_statistics)
    ;} 
}

# force/RotatedLinear.h
void wrap_rotated_linear() {
  {typedef RotatedLinear<T,2> Self;
  Class<Self>("RotatedLinear2d")
    .GEODE_INIT(NdArray<const T>,NdArray<const T>,NdArray<const T>)
    ;}

  {typedef RotatedLinear<T,3> Self;
  Class<Self>("RotatedLinear3d")
    .GEODE_INIT(NdArray<const T>,NdArray<const T>,NdArray<const T>)
    ;}
}

# force/EtherDrag
void wrap_ether_drag() {
  typedef real T;
  typedef Vector<T,3> TV;
  typedef EtherDrag<TV> Self;
  Class<Self>("EtherDrag")
    .GEODE_INIT(Array<const T>,T)
    ;
}

# force/ConstitutiveModel.h
void wrap_constitutive_model() {
  {typedef ConstitutiveModel<T,2> Self;
  Class<Self>("ConstitutiveModel2d");}

  {typedef ConstitutiveModel<T,3> Self;
  Class<Self>("ConstitutiveModel3d");}
}

# solver/brent.h
GEODE_FUNCTION(bracket)
GEODE_FUNCTION_2(brent,brent_py)

# solver/powell.h
GEODE_FUNCTION_2(powell,powell_py)

# svg/svg_to_bezier
void wrap_svg_to_bezier() {
  typedef SVGStyledPath Self;
  Class<Self>("SVGStyledPath")
    .GEODE_FIELD(fillColor)
    .GEODE_FIELD(strokeColor)
    .GEODE_FIELD(hasFill)
    .GEODE_FIELD(hasStroke)
    .GEODE_FIELD(CSSclass)
    .GEODE_FIELD(shapes)
    ;

  GEODE_FUNCTION(svgfile_to_styled_beziers)
  GEODE_FUNCTION(svgstring_to_styled_beziers)
  GEODE_FUNCTION(svgfile_to_beziers)
  GEODE_FUNCTION(svgstring_to_beziers)
}

# force/FiniteVolume
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

# random/counter.h
GEODE_FUNCTION(threefry)

# random/permute.h
void wrap_permute() {
  GEODE_FUNCTION(random_permute)
  GEODE_FUNCTION(random_unpermute)
}

# math/numeric_limits
static PyObject* build_limits(PyObject* object) {
  PyArray_Descr* dtype;
  if (!PyArray_DescrConverter(object,&dtype))
    return 0;
  const Ref<> save = steal_ref(*(PyObject*)dtype);
  const int type = dtype->type_num;
  switch (type) {
    case NumpyScalar<float>::value:  return to_python(new_<Limits<float>>());
    case NumpyScalar<double>::value: return to_python(new_<Limits<double>>());
    default:
      Ref<PyObject> s = steal_ref_check(PyObject_Str((PyObject*)dtype));
      throw TypeError(format("numeric_limits unimplemented for type %s",from_python<const char*>(s)));
  }
}

#endif
}
using namespace geode;
#ifdef GEODE_PYTHON
template<class T> static void wrap_helper() {
  typedef Limits<T> Self;
  Class<Self>("numeric_limits")
    .GEODE_FIELD(min)
    .GEODE_FIELD(max)
    .GEODE_FIELD(epsilon)
    .GEODE_FIELD(round_error)
    .GEODE_FIELD(infinity)
    .GEODE_FIELD(quiet_NaN)
    .GEODE_FIELD(signaling_NaN)
    .GEODE_FIELD(denorm_min)
    .GEODE_FIELD(digits)
    .GEODE_FIELD(digits10)
    .GEODE_FIELD(min_exponent)
    .GEODE_FIELD(min_exponent10)
    .GEODE_FIELD(max_exponent)
    .GEODE_FIELD(max_exponent10)
    .GEODE_REPR()
    ;
}
#endif

void wrap_numeric_limits() {
#ifdef GEODE_PYTHON
  wrap_helper<float>();
  wrap_helper<double>();
  GEODE_FUNCTION_2(numeric_limits,build_limits)
#endif
}

# random/sobol
template<int d> static void wrap_helper() {
  typedef Sobol<Vector<T,d>> Self;
  static char name[8] = "Sobol?d";
  name[5] = '0'+d;
  Class<Self>(name)
    .GEODE_INIT(Box<Vector<T,d>>)
    .GEODE_METHOD(vector)
    ;
}
void wrap_sobol() {
  wrap_helper<1>();
  wrap_helper<2>();
  wrap_helper<3>();
}

# mesh/PolygonSoup.h
void wrap_polygon_mesh() {
  typedef PolygonSoup Self;
  Class<Self>("PolygonSoup")
    .GEODE_INIT(Array<const int>,Array<const int>)
    .GEODE_FIELD(counts)
    .GEODE_FIELD(vertices)
    .GEODE_METHOD(segment_soup)
    .GEODE_METHOD(triangle_mesh)
    .GEODE_METHOD(nodes)
    ;
}

# math/optimal_sort.cpp
void wrap_optimal_sort() {
  GEODE_FUNCTION(optimal_sort_test)
  GEODE_FUNCTION(optimal_sort_stats)
}

# random/Random
void wrap_Random() {
  typedef Random Self;
  Class<Self>("Random")
    .GEODE_INIT(uint128_t)
    .GEODE_FIELD(seed)
    .GEODE_METHOD_2("normal",normal_py)
    .GEODE_METHOD_2("uniform",uniform_py)
    .GEODE_METHOD_2("uniform_int",uniform_int_py)
    ;

  GEODE_FUNCTION(random_bits_test)
}

# openmesh/decimate.h
void wrap_decimate() {
  GEODE_FUNCTION_2(decimate_py,decimate)
}

# openmesh/curvature.h
void wrap_curvature() {
  GEODE_FUNCTION(mean_curvatures)
  GEODE_FUNCTION(gaussian_curvatures)
}

# mesh/TriangleTopology.h
void wrap_corner_mesh() {
  #define SAFE_METHOD(name) GEODE_METHOD_2(#name,safe_##name)
  {
    typedef TriangleTopology Self;
    Class<Self>("TriangleTopology")
      .GEODE_INIT(const TriangleSoup&)
      .GEODE_METHOD(copy)
      .GEODE_METHOD(mutate)
      .GEODE_GET(n_vertices)
      .GEODE_GET(n_boundary_edges)
      .GEODE_GET(n_edges)
      .GEODE_GET(n_faces)
      .GEODE_GET(chi)
      .SAFE_METHOD(halfedge)
      .SAFE_METHOD(prev)
      .SAFE_METHOD(next)
      .SAFE_METHOD(src)
      .SAFE_METHOD(dst)
      .SAFE_METHOD(face)
      .SAFE_METHOD(left)
      .SAFE_METHOD(right)
      .SAFE_METHOD(face_vertices)
      .SAFE_METHOD(face_halfedges)
      .SAFE_METHOD(halfedge_vertices)
      .SAFE_METHOD(face_faces)
      .SAFE_METHOD(halfedge_faces)
      .SAFE_METHOD(outgoing)
      .SAFE_METHOD(incoming)
      .GEODE_METHOD(vertex_one_ring)
      .GEODE_METHOD(incident_faces)
      .GEODE_OVERLOADED_METHOD_2(HalfedgeId(Self::*)(VertexId, VertexId)const, "halfedge_between", halfedge)
      .GEODE_METHOD(common_halfedge)
      .GEODE_METHOD(elements)
      .GEODE_METHOD(degree)
      .GEODE_METHOD(has_boundary)
      .GEODE_METHOD(is_manifold)
      .GEODE_METHOD(is_manifold_with_boundary)
      .GEODE_METHOD(has_isolated_vertices)
      .GEODE_METHOD(boundary_loops)
      .GEODE_METHOD(assert_consistent)
      .GEODE_METHOD(dump_internals)
      .GEODE_METHOD(all_vertices)
      .GEODE_METHOD(all_faces)
      .GEODE_METHOD(all_halfedges)
      .GEODE_METHOD(all_boundary_edges)
      .GEODE_METHOD(all_interior_halfedges)
      .GEODE_OVERLOADED_METHOD(Range<TriangleTopologyIter<VertexId>>(Self::*)() const, vertices)
      .GEODE_OVERLOADED_METHOD(Range<TriangleTopologyIter<FaceId>>(Self::*)() const, faces)
      .GEODE_OVERLOADED_METHOD(Range<TriangleTopologyIter<HalfedgeId>>(Self::*)() const, halfedges)
      .GEODE_METHOD(boundary_edges)
      .GEODE_METHOD(interior_halfedges)
      .GEODE_METHOD(is_garbage_collected)
      ;
  }
  {
    typedef MutableTriangleTopology Self;
    Class<Self>("MutableTriangleTopology")
      .GEODE_INIT()
      .GEODE_METHOD(copy)
      .GEODE_METHOD(add_vertex)
      .GEODE_METHOD(add_vertices)
      .GEODE_METHOD(add_face)
      .GEODE_METHOD(add_faces)
      .SAFE_METHOD(erase_face)
      .SAFE_METHOD(erase_vertex)
      .SAFE_METHOD(erase_halfedge)
      .GEODE_METHOD(collect_garbage)
      .GEODE_METHOD(collect_boundary_garbage)
      #ifdef GEODE_PYTHON
      .GEODE_METHOD_2("add_vertex_field",add_vertex_field_py)
      .GEODE_METHOD_2("add_face_field",add_face_field_py)
      .GEODE_METHOD_2("add_halfedge_field",add_halfedge_field_py)
      .GEODE_METHOD_2("has_field",has_field_py)
      .GEODE_METHOD_2("remove_field",remove_field_py)
      .GEODE_METHOD_2("field",field_py)
      #endif
      .GEODE_METHOD(permute_vertices)
      ;
  }
  // For testing purposes
  GEODE_FUNCTION(corner_random_edge_flips)
  GEODE_FUNCTION(corner_random_face_splits)
  GEODE_FUNCTION(corner_mesh_destruction_test)

  GEODE_PYTHON_RANGE(TriangleTopologyIncoming, "IncomingHalfedgeIter")
  GEODE_PYTHON_RANGE(TriangleTopologyOutgoing, "OutgoingHalfedgeIter")
  GEODE_PYTHON_RANGE(TriangleTopologyIter<VertexId>, "SkippingVertexIter")
  GEODE_PYTHON_RANGE(TriangleTopologyIter<FaceId>, "SkippingFaceIter")
  GEODE_PYTHON_RANGE(TriangleTopologyIter<HalfedgeId>, "SkippingHalfedgeIter")
}

# openmesh/TriMesh
void wrap_trimesh() {
  typedef TriMesh Self;

  // need to specify exact type for overloaded functions
  typedef Box<Vector<real,3> > (TriMesh::*box_Method)() const;
  typedef TriMesh::FaceHandle (TriMesh::*fh_Method_vh_vh_vh)(TriMesh::VertexHandle, TriMesh::VertexHandle, TriMesh::Ver
  typedef Vector<TriMesh::VertexHandle, 3> (TriMesh::*Vvh3_Method_fh)(TriMesh::FaceHandle ) const;
  typedef Vector<TriMesh::VertexHandle, 2> (TriMesh::*Vvh2_Method_eh)(TriMesh::EdgeHandle ) const;
  
  typedef void (TriMesh::*v_Method_str)(string const &);
  typedef void (TriMesh::*v_CMethod_str)(string const &) const;

  typedef void (TriMesh::*v_Method_r_vec3)(real, const Vector<real, 3>&);
  typedef void (TriMesh::*v_Method_vec3_vec3)(Vector<real, 3>, const Vector<real, 3>&);

  typedef Ref<TriMesh> (TriMesh::*Mesh_CMethod_vfh)(vector<FaceHandle> const &faces) const;
  
  Class<Self>("TriMesh")
    .GEODE_INIT()
    .GEODE_METHOD(copy)
    .GEODE_METHOD(add_vertex)
    .GEODE_OVERLOADED_METHOD(fh_Method_vh_vh_vh, add_face)
    .GEODE_METHOD(add_vertices)
    .GEODE_METHOD(add_faces)
    .GEODE_METHOD(add_mesh)
    .GEODE_METHOD(n_vertices)
    .GEODE_METHOD(n_faces)
    .GEODE_METHOD(n_edges)
    .GEODE_METHOD(n_halfedges)
    .GEODE_METHOD(remove_infinite_vertices)
    .GEODE_OVERLOADED_METHOD(v_Method_str, read)
    .GEODE_OVERLOADED_METHOD(v_CMethod_str, write)
    .GEODE_METHOD(write_with_normals)
    .GEODE_OVERLOADED_METHOD(box_Method, bounding_box)
    .GEODE_METHOD(mean_edge_length)
    .GEODE_OVERLOADED_METHOD_2(Vvh3_Method_fh, "face_vertex_handles", vertex_handles)
    .GEODE_OVERLOADED_METHOD_2(Vvh2_Method_eh, "edge_vertex_handles", vertex_handles)
    .GEODE_METHOD(vertex_one_ring)
    .GEODE_METHOD(smooth_normal)
    .GEODE_METHOD(add_cylinder)
    .GEODE_METHOD(add_sphere)
    .GEODE_METHOD(add_box)
    .GEODE_METHOD(vertex_shortest_path)
    .GEODE_METHOD(elements)
    .GEODE_METHOD(invert)
    .GEODE_METHOD(to_vertex_handle)
    .GEODE_METHOD(from_vertex_handle)
    .GEODE_METHOD(select_faces)
    .GEODE_OVERLOADED_METHOD(Mesh_CMethod_vfh, extract_faces)
    .GEODE_METHOD_2("X",X_python)
    .GEODE_METHOD_2("set_X",set_X_python)
    .GEODE_METHOD(set_vertex_normals)
    .GEODE_METHOD(set_vertex_colors)
    .GEODE_METHOD(face_texcoords)
    .GEODE_METHOD(set_face_texcoords)
    .GEODE_METHOD(component_meshes)
    .GEODE_METHOD(largest_connected_component)
    .GEODE_METHOD(request_vertex_normals)
    .GEODE_METHOD(request_face_normals)
    .GEODE_METHOD(update_face_normals)
    .GEODE_METHOD(update_vertex_normals)
    .GEODE_METHOD(update_normals)
    .GEODE_METHOD(request_face_colors)
    .GEODE_METHOD(request_vertex_colors)
    .GEODE_METHOD_2("request_face_texcoords",request_halfedge_texcoords2D)
    .GEODE_METHOD(request_halfedge_texcoords2D)
    .GEODE_OVERLOADED_METHOD(real(Self::*)()const,volume)
    .GEODE_OVERLOADED_METHOD(real(Self::*)()const,area)
    .GEODE_OVERLOADED_METHOD_2(v_Method_r_vec3, "scale", scale)
    .GEODE_OVERLOADED_METHOD_2(v_Method_vec3_vec3, "scale_anisotropic", scale)
    .GEODE_OVERLOADED_METHOD(void(Self::*)(Matrix<real,4>const&),transform)
    .GEODE_METHOD(translate)
    .GEODE_METHOD(boundary_loops)
    .GEODE_METHOD(face_tree)
    .GEODE_METHOD(edge_tree)
    .GEODE_METHOD(point_tree)
    .GEODE_OVERLOADED_METHOD(OTriMesh::Point const &(Self::*)(VertexHandle)const, point)
    .GEODE_OVERLOADED_METHOD_2(OTriMesh::Point (Self::*)(FaceHandle,Vector<real,3>const&)const, "interpolated_point", p
    .GEODE_OVERLOADED_METHOD(Self::Normal (Self::*)(FaceHandle)const, normal)
    .GEODE_OVERLOADED_METHOD(Self::TV(Self::*)()const, centroid)
    .GEODE_OVERLOADED_METHOD_2(Self::TV(Self::*)(FaceHandle)const, "face_centroid", centroid)
    ;
}

# mesh/TriangleSubdivision
void wrap_triangle_subdivision() {
  typedef TriangleSubdivision Self;
  Class<Self>("TriangleSubdivision")
    .GEODE_INIT(TriangleSoup&)
    .GEODE_FIELD(coarse_mesh)
    .GEODE_FIELD(fine_mesh)
    .GEODE_FIELD(corners)
    .GEODE_METHOD_2("linear_subdivide",linear_subdivide_python)
    .GEODE_METHOD_2("loop_subdivide",loop_subdivide_python)
    ;
}

# mesh/ids.h
void wrap_ids() {
  GEODE_OBJECT(invalid_id);
  GEODE_OBJECT(erased_id);
  GEODE_OBJECT(vertex_position_id);
  GEODE_OBJECT(vertex_color_id);
  GEODE_OBJECT(vertex_texcoord_id);
  GEODE_OBJECT(face_color_id);
  GEODE_OBJECT(halfedge_color_id);
  GEODE_OBJECT(halfedge_texcoord_id);

  GEODE_PYTHON_RANGE(IdIter<VertexId>, "VertexIter")
  GEODE_PYTHON_RANGE(IdIter<FaceId>, "FaceIter")
  GEODE_PYTHON_RANGE(IdIter<HalfedgeId>, "HalfedgeIter")
}

# mesh/SegmentSoup.h
void wrap_segment_soup() {
  typedef SegmentSoup Self;
  Class<Self>("SegmentSoup")
    .GEODE_INIT(Array<const Vector<int,2> >)
    .GEODE_FIELD(d)
    .GEODE_FIELD(vertices)
    .GEODE_FIELD(elements)
    .GEODE_METHOD(segment_soup)
    .GEODE_METHOD(incident_elements)
    .GEODE_METHOD(adjacent_elements)
    .GEODE_METHOD(nodes)
    .GEODE_METHOD(neighbors)
    .GEODE_METHOD(element_normals)
    .GEODE_METHOD(nonmanifold_nodes)
    .GEODE_METHOD(polygons)
    .GEODE_METHOD(bending_tuples)
    ;
}

# mesh/HalfedgeMesh.h
void wrap_halfedge_mesh() {
  typedef HalfedgeMesh Self;
  Class<Self>("HalfedgeMesh")
    .GEODE_INIT()
    .GEODE_METHOD(copy)
    .GEODE_GET(n_vertices)
    .GEODE_GET(n_halfedges)
    .GEODE_GET(n_edges)
    .GEODE_GET(n_faces)
    .GEODE_GET(chi)
    .GEODE_METHOD(elements)
    .GEODE_METHOD(has_boundary)
    .GEODE_METHOD(is_manifold)
    .GEODE_METHOD(is_manifold_with_boundary)
    .GEODE_METHOD(boundary_loops)
    .GEODE_METHOD(add_vertex)
    .GEODE_METHOD(add_vertices)
    .GEODE_METHOD(add_face)
    .GEODE_METHOD(add_faces)
    .GEODE_METHOD(assert_consistent)
    .GEODE_METHOD(dump_internals)
    ;

  // For testing purposes
  GEODE_FUNCTION(halfedge_random_edge_flips)
  GEODE_FUNCTION(halfedge_random_face_splits)
  GEODE_FUNCTION(halfedge_mesh_destruction_test)
}

# mesh/TriangleSoup
void wrap_triangle_mesh() {
  typedef TriangleSoup Self;
  Class<Self>("TriangleSoup")
    .GEODE_INIT(Array<const Vector<int,3> >)
    .GEODE_FIELD(d)
    .GEODE_FIELD(elements)
    .GEODE_FIELD(vertices)
    .GEODE_METHOD(segment_soup)
    .GEODE_METHOD(triangle_mesh)
    .GEODE_METHOD(incident_elements)
    .GEODE_METHOD(adjacent_elements)
    .GEODE_METHOD(boundary_mesh)
    .GEODE_METHOD(bending_tuples)
    .GEODE_METHOD(nodes_touched)
    .GEODE_METHOD(area)
    .GEODE_METHOD(volume)
    .GEODE_METHOD(surface_area)
    .GEODE_METHOD(vertex_areas)
    .GEODE_METHOD(vertex_normals)
    .GEODE_METHOD(element_normals)
    .GEODE_METHOD(nodes)
    .GEODE_METHOD(nonmanifold_nodes)
    .GEODE_METHOD(sorted_neighbors)
    ;
}

# mesh/io.h
void wrap_mesh_io() {
  GEODE_FUNCTION(read_soup)
  GEODE_FUNCTION(read_polygon_soup)
  GEODE_FUNCTION(read_mesh)
  GEODE_FUNCTION_2(write_mesh,write_mesh_py)
}
'''
