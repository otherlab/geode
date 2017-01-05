//#####################################################################
// Class Vector
//#####################################################################
#pragma once

#include <geode/vector/Vector0d.h>
#include <geode/vector/Vector1d.h>
#include <geode/vector/Vector2d.h>
#include <geode/vector/Vector3d.h>
#include <geode/vector/Vector4d.h>
#include <geode/math/clamp.h>
#include <geode/math/inverse.h>
#include <geode/math/max.h>
#include <geode/math/min.h>
#include <geode/math/sqr.h>
#include <geode/python/forward.h>
#include <geode/python/repr.h>
#include <geode/python/to_python.h>
#include <geode/python/config.h>
#include <geode/utility/type_traits.h>
#include <cmath>
namespace geode {

using ::std::abs;
using ::std::floor;
using ::std::ceil;
using ::std::sqrt;
using ::std::exp;
using ::std::sin;
using ::std::cos;
using ::std::pow;

template<class TArray,class TIndices> class IndirectArray;

#ifdef GEODE_PYTHON
// Declare blanket to_python for numpy-incompatible vectors with inner types that can be converted
template<class T,int d> typename enable_if<has_to_python<T>, PyObject*>::type to_python(const Vector<T,d>& v);
#endif

// Declare the base set of numpy compatible vector conversions
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,2,int)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,int)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,4,int)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,2,long)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,long)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,4,long)

GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,2,short int)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,short int)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,4,short int)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,1,float)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,2,float)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,float)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,4,float)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,1,double)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,2,double)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,double)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,4,double)

template<class T,int d>
class Vector
{
    static_assert(d>4,"Smaller vector sizes are explicitly specialized");
    struct Unusable{};
public:
    typedef typename mpl::if_<IsScalar<T>,T,Unusable>::type Scalar;
    typedef T Element;
    typedef T value_type; // for stl
    typedef T* iterator; // for stl
    typedef const T* const_iterator; // for stl
    template<class> class result;
    template<class V> class result<V(int)>:public mpl::if_<is_const<V>,const T&,T&>{};
    static const int dimension = d;
    static const int m = d;
    static const bool is_const = false;

    T array[d];

    Vector()
    {
        for(int i=0;i<d;i++) array[i]=T();
    }

    Vector(const T& x0,const T& x1,const T& x2,const T& x3)
    {
        static_assert(d==4,"");array[0]=x0;array[1]=x1;array[2]=x2;array[3]=x3;
    }

    Vector(const T& x0,const T& x1,const T& x2,const T& x3,const T& x4)
    {
        static_assert(d==5,"");array[0]=x0;array[1]=x1;array[2]=x2;array[3]=x3;array[4]=x4;
    }

    Vector(const T& x0,const T& x1,const T& x2,const T& x3,const T& x4,const T& x5)
    {
        static_assert(d==6,"");array[0]=x0;array[1]=x1;array[2]=x2;array[3]=x3;array[4]=x4;array[5]=x5;
    }

    template<class T2,int d2>
    explicit Vector(const Vector<T2,d2>& v)
    {
        static_assert(d2<=d,"");
        for(int i=0;i<d2;i++) array[i]=T(v[i]);
        for(int i=d2;i<d;i++) array[i]=T();
    }

    template<class TVector>
    explicit Vector(const TVector& v)
    {
        static_assert(is_same<T,typename TVector::Element>::value && TVector::m==d,"");
        for(int i=0;i<d;i++) array[i]=v[i];
    }

    Vector(const Vector& v)
    {
        for(int i=0;i<d;i++) array[i]=v.array[i];
    }

    template<int n>
    Vector(const Vector<T,n>& v1,const Vector<T,d-n>& v2)
    {
        for(int i=0;i<n;i++) (*this)(i)=v1(i);for(int i=n;i<d;i++) (*this)(i)=v2(i-n);
    }

    template<class TVector> typename EnableForVectorLike<T,d,TVector,Vector&>::type
    operator=(const TVector& v)
    {
        for(int i=0;i<d;i++) array[i]=v[i];return *this;
    }

    Vector& operator=(const Vector& v)
    {
        for(int i=0;i<d;i++) array[i]=v.array[i];return *this;
    }

    constexpr int size() const
    {return m;}

    constexpr bool empty() const
    {return m>0;}

    const T& operator[](const int i) const
    {assert(unsigned(i)<d);return array[i];}

    T& operator[](const int i)
    {assert(unsigned(i)<d);return array[i];}

    T* data()
    {return array;}

    const T* data() const
    {return array;}

    template<class TIndices>
    IndirectArray<Vector,TIndices&> subset(const TIndices& indices)
    {return IndirectArray<Vector,TIndices&>(*this,indices);}

    template<class TIndices>
    IndirectArray<const Vector,TIndices&> subset(const TIndices& indices) const
    {return IndirectArray<const Vector,TIndices&>(*this,indices);}

    bool operator==(const Vector& v) const
    {for(int i=0;i<d;i++) if(array[i]!=v.array[i]) return false;return true;}

    bool operator!=(const Vector& v) const
    {return !((*this)==v);}

    Vector operator-() const
    {Vector r;for(int i=0;i<d;i++) r.array[i]=-array[i];return r;}

    Vector operator+(const T& a) const
    {Vector r;for(int i=0;i<d;i++) r.array[i]=array[i]+a;return r;}

    Vector& operator+=(const T& a)
    {for(int i=0;i<d;i++) array[i]+=a;return *this;}

    Vector operator-(const T& a) const
    {Vector r;for(int i=0;i<d;i++) r.array[i]=array[i]-a;return r;}

    Vector& operator-=(const T& a)
    {for(int i=0;i<d;i++) array[i]-=a;return *this;}

    Vector operator+(const Vector& v) const
    {Vector r;for(int i=0;i<d;i++) r.array[i]=array[i]+v.array[i];return r;}

    Vector& operator+=(const Vector& v)
    {for(int i=0;i<d;i++) array[i]+=v.array[i];return *this;}

    Vector operator-(const Vector& v) const
    {Vector r;for(int i=0;i<d;i++) r.array[i]=array[i]-v.array[i];return r;}

    Vector& operator-=(const Vector& v)
    {for(int i=0;i<d;i++) array[i]-=v.array[i];return *this;}

    Vector operator*(const T& a) const
    {Vector r;for(int i=0;i<d;i++) r.array[i]=array[i]*a;return r;}

    Vector& operator*=(const T& a)
    {for(int i=0;i<d;i++) array[i]*=a;return *this;}

    Vector operator>>(const int a)
    {Vector r; for(int i = 0; i < d; ++i) r.array[i] = array[i] >> a; return r;}

    Vector operator<<(const int a)
    {Vector r; for(int i = 0; i < d; ++i) r.array[i] = array[i] << a; return r;}

    Vector operator/(const T& a) const
    {return *this*(1/a);}

    Vector& operator/=(const T& a)
    {return *this*=1/a;}

    Vector operator*(const Vector& v) const
    {Vector r;for(int i=0;i<d;i++) r.array[i]=array[i]*v.array[i];return r;}

    Vector& operator*=(const Vector& v)
    {for(int i=0;i<d;i++) array[i]*=v.array[i];return *this;}

    Vector operator/(const Vector& v) const
    {Vector r;for(int i=0;i<d;i++) r.array[i]=array[i]/v.array[i];return r;}

    Vector& operator/=(const Vector& v)
    {for(int i=0;i<d;i++) array[i]/=v.array[i];return *this;}

    Vector operator*(const IntInverse<T> a) const
    {Vector r;for(int i=0;i<d;i++) r.array[i]=array[i]*a;return r;}

    Vector operator&(const T& a) const
    {Vector r;for(int i=0;i<d;i++) r.array[i]=array[i]&a;return r;}

    T sqr_magnitude() const
    {T r=0;for(int i=0;i<d;i++) r+=sqr(array[i]);return r;}

    T magnitude() const
    {return sqrt(sqr_magnitude());}

    T normalize()
    {T magnitude=magnitude();if(magnitude) *this*=1/magnitude;else *this=axis_vector(0);return magnitude;}

    Vector normalized() const
    {T magnitude=magnitude();if(magnitude) return *this*(1/magnitude);else return axis_vector(0);}

    T min() const
    {T r=array[0];for(int i=1;i<d;i++) r=geode::min(r,array[i]);return r;}

    T max() const
    {T r=array[0];for(int i=1;i<d;i++) r=geode::max(r,array[i]);return r;}

    bool elements_equal() const
    {bool equal=true;for(int i=0;i<d;i++) equal&=(array[i]==array[0]);return equal;}

    static Vector componentwise_min(const Vector& v1,const Vector& v2)
    {Vector r;for(int i=0;i<d;i++) r.array[i]=geode::min(v1.array[i],v2.array[i]);return r;}

    static Vector componentwise_max(const Vector& v1,const Vector& v2)
    {Vector r;for(int i=0;i<d;i++) r.array[i]=geode::max(v1.array[i],v2.array[i]);return r;}

    Vector projected_on_unit_direction(const Vector& direction) const
    {return dot(*this,direction)*direction;}

    Vector projected(const Vector& direction) const // un-normalized direction
    {return dot(*this,direction)/direction.sqr_magnitude()*direction;}

    void project_on_unit_direction(const Vector& direction)
    {*this=dot(*this,direction)*direction;}

    void project(const Vector& direction) // un-normalized direction
    {*this=dot(*this,direction)/direction.sqr_magnitude()*direction;}

    Vector projected_orthogonal_to_unit_direction(const Vector& direction) const
    {return *this-dot(*this,direction)*direction;}

    void project_orthogonal_to_unit_direction(const Vector& direction)
    {*this-=dot(*this,direction)*direction;}

    T sum() const
    {T r=array[0];for(int i=1;i<d;i++) r+=array[i];return r;}

    T average() const
    {return T(1./d)*sum();}

    T product() const
    {T r=array[0];for(int i=1;i<d;i++) r*=array[i];return r;}

    const Vector& column_sum() const
    {return *this;}

    int number_true() const
    {STATIC_ASSERT_SAME(T,bool);int count=0;for(int i=0;i<d;i++)if(array[i]) count++;return count;}

    static Vector axis_vector(const int axis)
    {Vector r;r[axis]=(T)1;return r;}

    static Vector ones()
    {Vector r;for(int i=0;i<d;i++) r.array[i]=(T)1;return r;}

    static Vector repeat(const T& constant)
    {Vector r;for(int i=0;i<d;i++) r.array[i]=constant;return r;}
    
    static Vector nans()
    {return Vector::repeat(std::numeric_limits<T>::quiet_NaN());}

    // shifts vector (wrapped) such that element a is first
    Vector<T,d> roll(int a) const {
      Vector<T,d> v;
      for (int i = 0; i < d; ++i) {
        v[i] = (*this)[(i+a) % d];
      }
      return v;
    }

    void fill(const T& constant)
    {for(int i=0;i<d;i++) array[i]=constant;}

    void get(T& element0,T& element1,T& element2,T& element3) const
    {static_assert(d==4,"");element0=array[0];element1=array[1];element2=array[2];element3=array[3];}

    void get(T& element0,T& element1,T& element2,T& element3,T& element4) const
    {static_assert(d==5,"");element0=array[0];element1=array[1];element2=array[2];element3=array[3];element4=array[4];}

    void get(T& element0,T& element1,T& element2,T& element3,T& element4,T& element5) const
    {static_assert(d==6,"");element0=array[0];element1=array[1];element2=array[2];element3=array[3];element4=array[4];element5=array[5];}

    void get(T& element0,T& element1,T& element2,T& element3,T& element4,T& element5,T& element6) const
    {static_assert(d==7,"");element0=array[0];element1=array[1];element2=array[2];element3=array[3];element4=array[4];element5=array[5];element6=array[6];}

    void get(T& element0,T& element1,T& element2,T& element3,T& element4,T& element5,T& element6,T& element7) const
    {static_assert(d==8,"");element0=array[0];element1=array[1];element2=array[2];element3=array[3];element4=array[4];element5=array[5];element6=array[6];element7=array[7];}

    void set(const T& element0,const T& element1,const T& element2,const T& element3)
    {static_assert(d==4,"");array[0]=element0;array[1]=element1;array[2]=element2;array[3]=element3;}

    void set(const T& element0,const T& element1,const T& element2,const T& element3,const T& element4)
    {static_assert(d==5,"");array[0]=element0;array[1]=element1;array[2]=element2;array[3]=element3;array[4]=element4;}

    void set(const T& element0,const T& element1,const T& element2,const T& element3,const T& element4,const T& element5)
    {static_assert(d==6,"");array[0]=element0;array[1]=element1;array[2]=element2;array[3]=element3;array[4]=element4;array[5]=element5;}

    void set(const T& element0,const T& element1,const T& element2,const T& element3,const T& element4,const T& element5,const T& element6)
    {static_assert(d==7,"");array[0]=element0;array[1]=element1;array[2]=element2;array[3]=element3;array[4]=element4;array[5]=element5;array[6]=element6;}

    void set(const T& element0,const T& element1,const T& element2,const T& element3,const T& element4,const T& element5,const T& element6,const T& element7)
    {static_assert(d==8,"");array[0]=element0;array[1]=element1;array[2]=element2;array[3]=element3;array[4]=element4;array[5]=element5;array[6]=element6;array[7]=element7;}

    template<class TFunction>
    static Vector map(const TFunction& f,const Vector& v)
    {Vector r;for(int i=0;i<d;i++) r.array[i]=f(v.array[i]);return r;}

    int find(const T& element) const
    {for(int i=0;i<d;i++) if(array[i]==element) return i;return -1;}

    bool contains(const T& element) const
    {for(int i=0;i<d;i++) if(array[i]==element) return true;return false;}

    template<class TArray>
    bool contains_all(const TArray& elements) const
    {STATIC_ASSERT_SAME(typename TArray::Element,T);
    for(int i=0;i<elements.size();i++) if(!contains(elements(i))) return false;
    return true;}

    template<class TArray>
    bool contains_any(const TArray& elements) const
    {STATIC_ASSERT_SAME(typename TArray::Element,T);
    for(int i=0;i<elements.size();i++) if(contains(elements(i))) return true;
    return false;}

    Vector<T,d-1> remove_index(const int index) const
    {assert(unsigned(index)<d);Vector<T,d-1> r;for(int i=0;i<d-1;i++) r[i]=(*this)[i+(i>=index)];return r;}

    Vector<T,d+1> insert(const T& element,const int index) const
    {Vector<T,d+1> r;r[index]=element;for(int i=0;i<d;i++) r[i+(i>=index)]=(*this)[i];return r;}

    Vector<T,d+1> append(const T& element) const
    {Vector<T,d+1> r;for(int i=0;i<d;i++) r[i]=(*this)[i];r[d]=element;return r;}

    template<int d2> Vector<T,d+d2> extend(const Vector<T,d2>& elements)
    {Vector<T,d+d2> r;
    for(int i=0;i<d;i++) r[i]=(*this)[i];
    for(int i=0;i<d2;i++) r[i+d]=elements[i];
    return r;}

    Vector<T,4> sorted() const
    {static_assert(d==4,"");Vector<T,4> r(*this);small_sort(r[0],r[1],r[2],r[3]);return r;}

    Vector reversed() const
    {Vector r;for(int i=0;i<d;i++) r.array[d-1-i]=array[i];return r;}

    template<int d1,int d2> Vector<T,d2-d1> slice() const
    {static_assert(0<=d1 && d1<=d2 && d2<=d,"");
    Vector<T,d2-d1> r;for(int i=d1;i<d2;i++) r[i-d1]=(*this)[i];return r;}

    template<int n> void split(Vector<T,n>& v1,Vector<T,d-n>& v2) const
    {for(int i=0;i<n;i++) v1(i)=(*this)(i);
    for(int i=n;i<d;i++) v2(i-n)=(*this)(i);}

    T* begin() // for stl
    {return array;}

    const T* begin() const // for stl
    {return array;}

    T* end() // for stl
    {return array+d;}

    const T* end() const // for stl
    {return array+d;}

    T& front() { return array[0]; }
    const T& front() const { return array[0]; }
    T& back() { return array[d-1]; }
    const T& back() const { return array[d-1]; }

//#####################################################################
};

typedef Vector<real, 2> Vector2;
typedef Vector<real, 3> Vector3;
typedef Vector<real, 4> Vector4;

//#####################################################################
// Miscellaneous free operators and functions
//#####################################################################
template<class T,int d> inline Vector<T,d>
operator+(const typename Hide<T>::type& a,const Vector<T,d>& v)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=a+v.array[i];return r;}

template<class T,int d> inline Vector<T,d>
operator-(const typename Hide<T>::type& a,const Vector<T,d>& v)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=a-v.array[i];return r;}

template<class T,int d> inline Vector<T,d>
operator*(const typename Hide<T>::type& a,const Vector<T,d>& v)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=a*v.array[i];return r;}

template<class T,int d> inline Vector<T,d>
operator/(const typename Hide<T>::type& a,const Vector<T,d>& v)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=a/v.array[i];return r;}

template<class T,int d> inline Vector<T,d>
abs(const Vector<T,d>& v)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=abs(v.array[i]);return r;}

template<class T,int d> inline Vector<T,d>
floor(const Vector<T,d>& v)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=floor(v.array[i]);return r;}

template<class T,int d> inline Vector<T,d>
ceil(const Vector<T,d>& v)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=ceil(v.array[i]);return r;}

template<class T,int d> inline Vector<T,d>
exp(const Vector<T,d>& v)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=exp(v.array[i]);return r;}

template<class T,int d> inline Vector<T,d>
sin(const Vector<T,d>& v)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=sin(v.array[i]);return r;}

template<class T,int d> inline Vector<T,d>
cos(const Vector<T,d>& v)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=cos(v.array[i]);return r;}

template<class T,int d> inline Vector<T,d>
sqrt(const Vector<T,d>& v)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=sqrt(v.array[i]);return r;}

template<class T,int d> inline Vector<T,d>
inverse(const Vector<T,d>& v)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=1/v.array[i];return r;}

template<class T, int d> auto
sign(const Vector<T, d> v) -> Vector<typename remove_const_reference<decltype(sign(v[0]))>::type,d> {
  Vector<typename remove_const_reference<decltype(sign(v[0]))>::type,d> result;
  for(int i = 0; i < d; ++i)
    result[i] = sign(v[i]);
  return result;
}

template<class T,int d>
inline Vector<T,d> wrap(const Vector<T,d>& value,const Vector<T,d>& lower,const Vector<T,d>& upper)
{Vector<T,d> result;for(int i=0;i<d;i++) result(i)=wrap(value(i),lower(i),upper(i));return result;}

template<class T,int d>
inline bool isfinite(const Vector<T,d>& v)
{for(int i=0;i<d;i++) if(!isfinite(v[i])) return false;return true;}

//#####################################################################
// Functions clamp, clamp_min, clamp_max, in_bounds
//#####################################################################
template<class T,int d> inline Vector<T,d>
clamp(const Vector<T,d>& v,const Vector<T,d>& vmin,const Vector<T,d>& vmax)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=clamp(v.array[i],vmin.array[i],vmax.array[i]);return r;}

template<class T,int d> inline Vector<T,d>
clamp(const Vector<T,d>& v,const T& min,const T& max)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=clamp(v.array[i],min,max);return r;}

template<class T,int d> inline Vector<T,d>
clamp_min(const Vector<T,d>& v,const Vector<T,d>& vmin)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=clamp_min(v.array[i],vmin.array[i]);return r;}

template<class T,int d> inline Vector<T,d>
clamp_min(const Vector<T,d>& v,const T& min)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=clamp_min(v.array[i],min);return r;}

template<class T,int d> inline Vector<T,d>
clamp_max(const Vector<T,d>& v,const Vector<T,d>& vmax)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=clamp_max(v.array[i],vmax.array[i]);return r;}

template<class T,int d> inline Vector<T,d>
clamp_max(const Vector<T,d>& v,const T& max)
{Vector<T,d> r;for(int i=0;i<d;i++) r.array[i]=clamp_max(v.array[i],max);return r;}

template<class T,int d> inline Vector<T,d>
in_bounds(const Vector<T,d>& v,const Vector<T,d>& vmin,const Vector<T,d>& vmax)
{for(int i=0;i<d;i++) if(!in_bounds(v.array[i],vmin.array[i],vmax.array[i])) return false;
return true;}

//#####################################################################
// Stream input and output
//#####################################################################
template<class T,int d> inline std::istream&
operator>>(std::istream& input,Vector<T,d>& v)
{input>>expect('[');if(d) input>>v[0];for(int i=1;i<d;i++) input>>expect(',')>>v[i];return input>>expect(']');}

template<class T,int d> inline std::ostream&
operator<<(std::ostream& output,const Vector<T,d>& v)
{output<<'[';if(d) output<<v[0];for(int i=1;i<d;i++) output<<","<<v[i];output<<']';return output;}

template<int d> inline std::ostream&
operator<<(std::ostream& output,const Vector<unsigned char,d>&v)
{output<<'[';if(d) output<<(int)v[0];for(int i=1;i<d;i++) output<<","<<(int)v[i];output<<']';return output;}

template<class T> string
tuple_repr(const Vector<T,1>& v)
{return format("(%s,)",repr(v[0]));}

template<class T> string
tuple_repr(const Vector<T,2>& v)
{return format("(%s,%s)",repr(v[0]),repr(v[1]));}

template<class T> string
tuple_repr(const Vector<T,3>& v)
{return format("(%s,%s,%s)",repr(v[0]),repr(v[1]),repr(v[2]));}

template<class T> string
tuple_repr(const Vector<T,4>& v)
{return format("(%s,%s,%s,%s)",repr(v[0]),repr(v[1]),repr(v[2]),repr(v[3]));}

//#####################################################################
// Vector construction
//#####################################################################
#ifdef GEODE_VARIADIC

template<class T,class... Args> static inline auto vec(const Args&... args)
  -> Vector<T,sizeof...(Args)> {
  return Vector<T,sizeof...(Args)>(args...);
}

template<class... Args> static inline auto vec(const Args&... args)
  -> Vector<typename common_type<Args...>::type,sizeof...(Args)> {
  return Vector<typename common_type<Args...>::type,sizeof...(Args)>(args...);
}

#else

template<class T> static inline Vector<T,0> vec() { return Vector<T,0>(); }
template<class T> static inline Vector<T,1> vec(const T& a0) { return Vector<T,1>(a0); }
template<class T> static inline Vector<T,2> vec(const T& a0,const T& a1) { return Vector<T,2>(a0,a1); }
template<class T> static inline Vector<T,3> vec(const T& a0,const T& a1,const T& a2) { return Vector<T,3>(a0,a1,a2); }
template<class T> static inline Vector<T,4> vec(const T& a0,const T& a1,const T& a2,const T& a3) { return Vector<T,4>(a0,a1,a2,a3); }
template<class T> static inline Vector<T,5> vec(const T& a0,const T& a1,const T& a2,const T& a3,const T& a4) { return Vector<T,5>(a0,a1,a2,a3,a4); }
template<class T> static inline Vector<T,6> vec(const T& a0,const T& a1,const T& a2,const T& a3,const T& a4,const T& a5) { return Vector<T,6>(a0,a1,a2,a3,a4,a5); }

#endif
//#####################################################################

template<class T,int d> const int Vector<T,d>::dimension;
template<class T,int d> const int Vector<T,d>::m;

}
