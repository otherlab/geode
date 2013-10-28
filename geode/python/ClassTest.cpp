//#####################################################################
// Class ClassTest
//#####################################################################
//
// Tests various python wrapping features.
//
//#####################################################################
#include <geode/python/Class.h>
#include <geode/python/forward.h>
#include <geode/python/Object.h>
#include <geode/python/Ptr.h>
#include <geode/python/Ref.h>
namespace geode {
namespace {

class ClassTest : public Object {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef Object Base;

  int field;
  int attr;
  Ref<Object> ref;
  Ptr<Object> ptr;
  Ref<PyObject> ref2;
  Ptr<PyObject> ptr2;
  static const int static_const = 17;
private:
  int data_;

protected:
  ClassTest(Ref<Object> ref)
    : field(0), attr(8), ref(ref), ref2(ref), data_(0) {}
public:
  virtual ~ClassTest() {}

  int normal(int x) {return 2*x;}
  virtual int virtual_(int x) {return 3*x;}
  static int static_(int x) {return 5*x;}
  int operator()(int x) {return 4*x;}
  int prop() const {return 17;}
  int data() const {return data_;}
  void set_data(int d) {data_=d;}

  int getattr(const string& name) {
    if (name=="attr")
      return attr;
    throw AttributeError("getattr");
  }

  void setattr(const string& name, PyObject* value) {
#ifdef GEODE_PYTHON
    if (name=="attr")
      attr = from_python<int>(value);
    else
      throw AttributeError("setattr");
#endif
  }
};

const int ClassTest::static_const;

GEODE_DEFINE_TYPE(ClassTest)

}
}
using namespace geode;

void wrap_test_class() {
  // Make sure the class is usable from C++ before python initialization
  {
    Ref<ClassTest> self(new_<ClassTest>(new_<Object>()));
    self->normal(0);
    self->virtual_(0);
  } // including destruction

  typedef ClassTest Self;
  Class<Self>("ClassTest")
    .GEODE_INIT(Ref<Object>)
    .GEODE_FIELD(field)
    .GEODE_FIELD(ref)
    .GEODE_FIELD(ptr)
    .GEODE_FIELD(ref2)
    .GEODE_FIELD(ptr2)
    .GEODE_FIELD(static_const)
    .GEODE_METHOD(normal)
    .GEODE_METHOD(virtual_)
    .GEODE_METHOD(static_)
    .GEODE_CALL(int,int)
    .GEODE_GET(prop)
    .GEODE_GETSET(data)
    .getattr()
    .setattr()
    ;
}
