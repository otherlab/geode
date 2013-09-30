//#####################################################################
// Class Vector0d
//#####################################################################
#pragma once

#include <other/core/vector/ScalarPolicy.h>
#include <other/core/math/hash.h>
#include <other/core/utility/debug.h>
#include <other/core/utility/stream.h>
#include <cmath>
#include <cassert>
#include <iostream>
namespace other {

template<class T>
class Vector<T,0>
{
    struct Unusable{};
public:
    enum Workaround1 {dimension=0};
    enum Workaround2 {m=0};
    typedef typename mpl::if_<IsScalar<T>,T,Unusable>::type Scalar;
    typedef T Element;

    Vector()
    {}

    template<class T2> explicit
    Vector(const Vector<T2,0>& vector_input)
    {}

    const T& operator[](const int) const
    {OTHER_FATAL_ERROR();}

    T& operator[](const int)
    {OTHER_FATAL_ERROR();}

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

}
