//#####################################################################
// Class ClassTest
//#####################################################################
//
// Tests various python wrapping features.
//
//#####################################################################
#include <other/core/python/Class.h>
#include <other/core/python/forward.h>
#include <other/core/python/Object.h>
#include <other/core/python/Ptr.h>
#include <other/core/python/Ref.h>
namespace other {
namespace {

class ClassTest : public Object {
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
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
#ifdef OTHER_PYTHON
    if (name=="attr")
      attr = from_python<int>(value);
    else
      throw AttributeError("setattr");
#endif
  }
};

const int ClassTest::static_const;

OTHER_DEFINE_TYPE(ClassTest)

}
}
using namespace other;

void wrap_test_class() {
  // Make sure the class is usable from C++ before python initialization
  {
    Ref<ClassTest> self(new_<ClassTest>(new_<Object>()));
    self->normal(0);
    self->virtual_(0);
  } // including destruction

  typedef ClassTest Self;
  Class<Self>("ClassTest")
    .OTHER_INIT(Ref<Object>)
    .OTHER_FIELD(field)
    .OTHER_FIELD(ref)
    .OTHER_FIELD(ptr)
    .OTHER_FIELD(ref2)
    .OTHER_FIELD(ptr2)
    .OTHER_FIELD(static_const)
    .OTHER_METHOD(normal)
    .OTHER_METHOD(virtual_)
    .OTHER_METHOD(static_)
    .OTHER_CALL(int,int)
    .OTHER_GET(prop)
    .OTHER_GETSET(data)
    .getattr()
    .setattr()
    ;
}
