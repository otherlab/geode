//#####################################################################
// Class NdArray
//#####################################################################
#include <geode/array/convert.h>
namespace geode {

NDARRAY_CONVERSIONS(int)
NDARRAY_CONVERSIONS(unsigned long)
NDARRAY_CONVERSIONS(unsigned long long)
NDARRAY_CONVERSIONS(float)
NDARRAY_CONVERSIONS(double)
NDARRAY_CONVERSIONS(Vector<real,2>)
NDARRAY_CONVERSIONS(Vector<real,3>)

}
