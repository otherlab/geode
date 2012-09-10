//#####################################################################
// Class NdArray
//#####################################################################
#include <other/core/array/convert.h>
namespace other {

NDARRAY_CONVERSIONS(int)
NDARRAY_CONVERSIONS(unsigned long)
NDARRAY_CONVERSIONS(unsigned long long)
NDARRAY_CONVERSIONS(float)
NDARRAY_CONVERSIONS(double)
NDARRAY_CONVERSIONS(Vector<real,3>)

}
