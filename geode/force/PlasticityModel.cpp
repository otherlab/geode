//#####################################################################
// Class PlasticityModel
//##################################################################### 
#include <geode/force/PlasticityModel.h>
#include <geode/python/Class.h>
#include <geode/vector/SymmetricMatrix.h>
namespace geode {

typedef real T;
template<> GEODE_DEFINE_TYPE(PlasticityModel<T,2>)
template<> GEODE_DEFINE_TYPE(PlasticityModel<T,3>)

}
using namespace geode;

void wrap_plasticity_model() {
  {typedef PlasticityModel<T,2> Self;
  Class<Self>("PlasticityModel2d");}

  {typedef PlasticityModel<T,3> Self;
  Class<Self>("PlasticityModel3d");}
}
