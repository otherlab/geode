//#####################################################################
// Class Array
//#####################################################################
#include <geode/array/convert.h>
#include <geode/vector/Vector.h>
namespace geode {

void out_of_bounds(const type_info& type, const int size, const int i) {
  throw IndexError(format("array index out of bounds: type %s, len %d, index %d",type.name(),size,i));
}

ARRAY_CONVERSIONS(1,bool)
ARRAY_CONVERSIONS(1,char)
ARRAY_CONVERSIONS(1,uint8_t)
ARRAY_CONVERSIONS(1,uint16_t)
ARRAY_CONVERSIONS(1,int)
ARRAY_CONVERSIONS(1,int64_t)
ARRAY_CONVERSIONS(1,uint64_t)
ARRAY_CONVERSIONS(1,Vector<int,2>)
ARRAY_CONVERSIONS(1,Vector<int,3>)
ARRAY_CONVERSIONS(1,Vector<int,4>)
ARRAY_CONVERSIONS(1,Vector<int,8>)
ARRAY_CONVERSIONS(1,float)
ARRAY_CONVERSIONS(1,double)
ARRAY_CONVERSIONS(2,int)
ARRAY_CONVERSIONS(2,float)
ARRAY_CONVERSIONS(2,double)
ARRAY_CONVERSIONS(2,uint8_t)
ARRAY_CONVERSIONS(2,unsigned long)
ARRAY_CONVERSIONS(2,uint64_t)
ARRAY_CONVERSIONS(3,float)
ARRAY_CONVERSIONS(3,double)
ARRAY_CONVERSIONS(3,uint8_t)
ARRAY_CONVERSIONS(1,long)
ARRAY_CONVERSIONS(1,Vector<long,2>)
ARRAY_CONVERSIONS(1,Vector<long,3>)
ARRAY_CONVERSIONS(1,Vector<long,4>)
ARRAY_CONVERSIONS(1,Vector<float,2>)
ARRAY_CONVERSIONS(1,Vector<double,2>)
ARRAY_CONVERSIONS(2,Vector<real,2>)
ARRAY_CONVERSIONS(1,Vector<float,3>)
ARRAY_CONVERSIONS(1,Vector<float,4>)
ARRAY_CONVERSIONS(1,Vector<double,3>)
ARRAY_CONVERSIONS(1,Vector<double,4>)
ARRAY_CONVERSIONS(2,Vector<float,3>)
ARRAY_CONVERSIONS(2,Vector<double,3>)
ARRAY_CONVERSIONS(1,Vector<Vector<real,2>,3>)
ARRAY_CONVERSIONS(2,Vector<unsigned char,3>)
ARRAY_CONVERSIONS(2,Vector<unsigned char,4>)
ARRAY_CONVERSIONS(2,Vector<float,4>)
ARRAY_CONVERSIONS(2,Vector<double,4>)

}
