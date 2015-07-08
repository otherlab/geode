//#####################################################################
// Class Vector0d
//#####################################################################
#pragma once

#include <geode/vector/ScalarPolicy.h>
#include <geode/math/hash.h>
#include <geode/utility/debug.h>
#include <geode/utility/stream.h>
#include <cmath>
#include <cassert>
#include <iostream>
namespace geode {

template<class T>
class Vector<T,0>
{
    struct Unusable{};
public:
    static const int dimension = 0;
    static const int m = 0;
    typedef typename mpl::if_<IsScalar<T>,T,Unusable>::type Scalar;
    typedef T Element;

    constexpr Vector() = default;

    template<class T2> explicit constexpr
    Vector(const Vector<T2,0>& vector_input)
    {}

    constexpr int size() const
    {return 0;}

    constexpr bool empty() const
    {return true;}

    const T& operator[](const int) const
    {GEODE_FATAL_ERROR();}

    T& operator[](const int)
    {GEODE_FATAL_ERROR();}

    T* data()
    {return 0;}

    const T* data() const
    {return 0;}

    template<class TIndices>
    IndirectArray<Vector,TIndices&> subset(const TIndices& indices)
    {return IndirectArray<Vector,TIndices&>(*this,indices);}

    template<class TIndices>
    IndirectArray<const Vector,TIndices&> subset(const TIndices& indices) const
    {return IndirectArray<const Vector,TIndices&>(*this,indices);}

    bool operator==(const Vector& v) const
    {return true;}

    bool operator!=(const Vector& v) const
    {return false;}

    Vector operator-() const
    {return *this;}

    Vector& operator+=(const Vector&)
    {return *this;}

    Vector& operator-=(const Vector&)
    {return *this;}

    Vector& operator*=(const Vector&)
    {return *this;}

    Vector& operator*=(const T&)
    {return *this;}

    Vector& operator/=(const T&)
    {return *this;}

    Vector& operator/=(const Vector&)
    {return *this;}

    Vector operator+(const Vector&) const
    {return *this;}

    Vector operator+(const T&) const
    {return *this;}

    Vector operator-(const Vector&) const
    {return *this;}

    Vector operator*(const Vector&) const
    {return *this;}

    Vector operator/(const Vector&) const
    {return *this;}

    Vector operator*(const T&) const
    {return *this;}

    Vector operator/(const T&) const
    {return *this;}

    Vector operator&(const T& a) const
    {return *this;}

    bool contains(const T&) const
    {return false;}

    T sqr_magnitude() const
    {return 0;}

    T magnitude() const
    {return 0;}

    T normalize()
    {return T();}

    Vector normalized() const
    {return *this;}

    Vector<T,1> insert(const T& element,const int index) const
    {Vector<T,1> r;r[index]=element;return r;}

    static Vector ones()
    {return Vector();}

    static Vector repeat(const T& constant)
    {return Vector();}
    
    static Vector nans()
    {return Vector();}

    // For stl
    T* begin() { return 0; }
    T* end() { return 0; }
    const T* begin() const { return 0; }
    const T* end() const { return 0; }

    Vector sorted() const
    {return *this;}

    template<class RW>
    void read(std::istream&)
    {}

    template<class RW>
    void write(std::ostream&) const
    {}
};

template<class T> inline Vector<T,0>
operator*(const typename Hide<T>::type&,const Vector<T,0>& v)
{return v;}

template<class T> inline Vector<T,0>
operator/(const typename Hide<T>::type&,const Vector<T,0>& v)
{return v;}

template<class T> inline T
dot(const Vector<T,0>&,const Vector<T,0>&)
{return T();}

template<class T> inline Hash
hash_reduce(const Vector<T,0>& key)
{return Hash();}

template<class T> const int Vector<T,0>::dimension;
template<class T> const int Vector<T,0>::m;

}
