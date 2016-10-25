//#####################################################################
// Class Array
//#####################################################################
#include <geode/array/convert.h>
#include <geode/vector/Vector.h>
namespace geode {

void out_of_bounds(const type_info& type, const int size, const int i) {
  throw IndexError(format("array index out of bounds: type %s, len %d, index %d",type.name(),size,i));
}

// Depending on the platform, primitive integer types (i.g. int) and fixed width integer types (i.g. int64_t) might or not be the same type
// Comparing sizes or limits isn't enough (i.g. if 'int' and 'long' are both 64 bits, 'int64_t' could be 'long' or 'int')
// This makes it difficult to instantiate conversions of both primitive and fixed width types without getting "duplicate explicit instantiation" errors
// To avoid this, conversions for fixed width types are instantiated for a dummy type if it duplicates a primitive type already in use
// It would be better to not instantiate anything, but I don't know a way to do that
namespace {
  // Helper classes to check is_same against multiple types
  template<bool... Args> struct template_any : public std::false_type { };
  template<bool A0, bool... Args> struct template_any<A0, Args...> : public std::integral_constant<bool, A0 || template_any<Args...>::value> { };
  template<class T, class... Args> using any_same = std::integral_constant<bool, template_any<std::is_same<T,Args>::value...>::value>;

  // We use this class to wrap duplicate types so that conversion functions are unique
  // I think it should be impossible to ever refer to these conversion functions except possibly when looking at a symbol dump
  template<class T, size_t N> struct UniqueDummyWrapper {};
  // We define these specializations so that conversion functions can successfully be instantiated
  template<class T, size_t N> struct geode::NumpyIsScalar<UniqueDummyWrapper<T, N>>:public mpl::true_{};
  template<class T, size_t N> struct geode::NumpyScalar<UniqueDummyWrapper<T, N>>:public NumpyScalar<T>{};
}

// ENABLE_IF_UNIQUE can be used to avoid duplicate explicit template instantiation in ARRAY_CONVERSIONS
// It maps a type T to either itself or 'UniqueDummyWrapper<T, __LINE__>' if it matches any of the other types passed
#define ENABLE_IF_UNIQUE(T, ...) typename std::conditional<any_same<T, __VA_ARGS__>::value, UniqueDummyWrapper<T, __LINE__>, T>::type

// Non integer primitive types
ARRAY_CONVERSIONS(1,bool)
ARRAY_CONVERSIONS(1,char)
// Primitive signed integer types
// Signed fixed width conversions of the same dimension need to ensure they don't overlap any of these types
ARRAY_CONVERSIONS(1,int)
ARRAY_CONVERSIONS(1,long)
ARRAY_CONVERSIONS(2,int)
// Primitive unsigned integer types
// Unsigned fixed width conversions of the same dimension need to ensure they don't overlap any of these types
ARRAY_CONVERSIONS(2,unsigned long)
// Signed fixed width integer types
ARRAY_CONVERSIONS(1,ENABLE_IF_UNIQUE(int64_t, int, long))
// Unsigned fixed width integer types
ARRAY_CONVERSIONS(1,uint8_t)
ARRAY_CONVERSIONS(1,uint16_t)
ARRAY_CONVERSIONS(1,uint64_t)
ARRAY_CONVERSIONS(2,uint8_t)
ARRAY_CONVERSIONS(2,ENABLE_IF_UNIQUE(uint64_t, unsigned long))
ARRAY_CONVERSIONS(3,uint8_t)
// Floating point types
ARRAY_CONVERSIONS(1,float)
ARRAY_CONVERSIONS(1,double)
ARRAY_CONVERSIONS(2,float)
ARRAY_CONVERSIONS(2,double)
ARRAY_CONVERSIONS(3,float)
ARRAY_CONVERSIONS(3,double)
// Arrays of Vectors
ARRAY_CONVERSIONS(1,Vector<int,2>)
ARRAY_CONVERSIONS(1,Vector<int,3>)
ARRAY_CONVERSIONS(1,Vector<int,4>)
ARRAY_CONVERSIONS(1,Vector<int,8>)
ARRAY_CONVERSIONS(1,Vector<long,2>)
ARRAY_CONVERSIONS(1,Vector<long,3>)
ARRAY_CONVERSIONS(1,Vector<long,4>)
ARRAY_CONVERSIONS(1,Vector<float,2>)
ARRAY_CONVERSIONS(1,Vector<float,3>)
ARRAY_CONVERSIONS(1,Vector<float,4>)
ARRAY_CONVERSIONS(1,Vector<double,2>)
ARRAY_CONVERSIONS(1,Vector<double,3>)
ARRAY_CONVERSIONS(1,Vector<double,4>)
ARRAY_CONVERSIONS(1,Vector<Vector<real,2>,3>)
ARRAY_CONVERSIONS(2,Vector<real,2>)
ARRAY_CONVERSIONS(2,Vector<float,3>)
ARRAY_CONVERSIONS(2,Vector<double,3>)
ARRAY_CONVERSIONS(2,Vector<unsigned char,3>)
ARRAY_CONVERSIONS(2,Vector<unsigned char,4>)
ARRAY_CONVERSIONS(2,Vector<float,4>)
ARRAY_CONVERSIONS(2,Vector<double,4>)

}
