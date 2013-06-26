//#####################################################################
// Class PlasticityModel
//##################################################################### 
#include <other/core/force/PlasticityModel.h>
#include <other/core/python/Class.h>
#include <other/core/vector/SymmetricMatrix.h>
namespace other {

typedef real T;
template<> OTHER_DEFINE_TYPE(PlasticityModel<T,2>)
template<> OTHER_DEFINE_TYPE(PlasticityModel<T,3>)

}
using namespace other;

void wrap_plasticity_model() {
  {typedef PlasticityModel<T,2> Self;
  Class<Self>("PlasticityModel2d");}

  {typedef PlasticityModel<T,3> Self;
  Class<Self>("PlasticityModel3d");}
}
