//#####################################################################
// Class PlasticityModel
//##################################################################### 
#pragma once

#include <other/core/array/Array.h>
namespace other{

template<class T,int d>
class PlasticityModel:public Object
{
public:
    OTHER_DECLARE_TYPE
    typedef Object Base;

    Array<SymmetricMatrix<T,d> > Fp_inverse;

    PlasticityModel(const int elements)
    {
        Fp_inverse.exact_resize(elements);
        Fp_inverse.fill(SymmetricMatrix<T,d>::identity_matrix());
    }
    
    virtual ~PlasticityModel()
    {}

    virtual bool project_Fe(const DiagonalMatrix<T,d>& Fe_trial,DiagonalMatrix<T,d>& Fe_project) const=0;
    virtual void project_Fp(const int tetrahedron,const Matrix<T,d>& Fp_trial)=0;
};
}
