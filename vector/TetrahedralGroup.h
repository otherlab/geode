//#####################################################################
// Class TetrahedralGroup
//#####################################################################
#pragma once

#include <other/core/vector/Rotation.h>
namespace other {

template<class T>
class TetrahedralGroup
{
public:
    int g; // group element (0 to 11)
private:
    static const Rotation<Vector<T,3> > quaternions[12];
    static const Rotation<Vector<int,3> > quaternions_times_2[12];
    static const int multiplication_table[12][12];
    static const int inversion_table[12];
public:
    static const TetrahedralGroup<T> e,i,j,k,a,a2,b,b2,c,c2,d,d2; // letter names for elements

    TetrahedralGroup(const int g_input=0)
        :g(g_input)
    {
        assert(unsigned(g)<12);
    }

    TetrahedralGroup<T> operator*(const TetrahedralGroup<T> g2) const
    {return multiplication_table[g][g2.g];}

    TetrahedralGroup<T> inverse() const
    {return inversion_table[g];}

    Vector<T,3> operator*(const Vector<T,3>& v) const
    {return quaternions[g]*v;}

    Vector<int,3> operator*(const Vector<int,3>& v) const
    {return quaternions_times_2[g]*v/4;}

    Vector<T,3> inverse_times(const Vector<T,3>& v) const
    {return quaternions[g].inverse_times(v);}

    Vector<int,3> inverse_times(const Vector<int,3>& v) const
    {return quaternions_times_2[g].inverse_times(v)/4;}

    static TetrahedralGroup<T> cyclic_shift_axes(const int times=1)
    {switch(times%3){case 0:return e;case 1:return a2;default:return a;}}

    static void assert_correctness()
    {for(int g=0;g<12;g++)assert(!multiplication_table[g][inversion_table[g]] && !multiplication_table[inversion_table[g]][g]);
    for(int g=0;g<12;g++)for(int h=0;h<12;h++){
        Rotation<Vector<T,3> > gh = quaternions[multiplication_table[g][h]],g_times_h=quaternions[g]*quaternions[h];
        OTHER_ASSERT(gh==g_times_h);}}
};
}
