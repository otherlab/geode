#include <geode/array/Array2d.h>
#include <geode/vector/Vector2d.h>
#include <geode/array/stencil.h>
#include <geode/python/wrap.h>
#include <geode/python/function.h>
#include <geode/python/Class.h>
#include <geode/python/from_python.h>
#include <geode/python/to_python.h>
namespace geode {

template<> GEODE_DEFINE_TYPE(MaxStencil<int>)
template<> GEODE_DEFINE_TYPE(MaxStencil<float>)
template<> GEODE_DEFINE_TYPE(MaxStencil<double>)
template<> GEODE_DEFINE_TYPE(MaxStencil<uint8_t>)

}
using namespace geode;

void wrap_stencil() {
  typedef uint8_t T;
  typedef const function<T (const Array<T,2>, Vector<int,2> const &)> ftype;
  typedef void(*stencil_ftype)(ftype &, int, const Array<T,2>);
  GEODE_FUNCTION_2(apply_stencil_uint8, static_cast<stencil_ftype>(apply_stencil<ftype, T, 2>));
  typedef MaxStencil<T> Self;
  Class<Self>("MaxStencil_uint8")
    .GEODE_INIT(int)
    .GEODE_FIELD(r)
    .GEODE_CALL(const Array<const typename Self::value_type,2>, Vector<int,2> const &)
    ;
}

