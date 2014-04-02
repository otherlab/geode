// Object: A base class for reference counted C++ objects.

#include <geode/utility/Object.h>
#include <geode/utility/format.h>
namespace geode {

Object::Object() {}

Owner::~Owner() {}

string Object::repr() const {
  return format("<object of type %s>",typeid(*this).name());
}

}
