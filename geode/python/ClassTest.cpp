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

class ClassTest : public Object, public WeakRefSupport {
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

  void dump_mem() const {
    #ifdef GEODE_PYTHON
    std::cout << "dumping ClassTest object at " << this << " (0 is at start of PyObject)" << std::endl;
    Object const *othis = dynamic_cast<Object const*>(this);
    WeakRefSupport const *wthis = dynamic_cast<WeakRefSupport const *>(this);
    PyObject const *pythis = to_python(geode::ref(*this));

    std::cout << "  Object instance at " << othis << std::endl;
    std::cout << "  WeakRefSupport instance at " << wthis << std::endl;
    std::cout << "  PyObject at " << pythis << std::endl;

    if (pythis != 0) {
      uint8_t const *p = (uint8_t const *)pythis;
      for (int i = -24; i < (int)(sizeof(ClassTest) + sizeof(PyObject) + pytype.tp_weaklistoffset); i += 8) {
        std::cout << format("0x%016x (%4d): %02x %02x %02x %02x  %02x %02x %02x %02x",
                            p+i, i, p[i], p[i+1], p[i+2], p[i+3], p[i+4], p[i+5], p[i+6], p[i+7]) << std::endl;
      }
    }
    #endif
  }

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

  GEODE_UNUSED friend int hash_reduce(const ClassTest& self) {
    return self.field;
  }

  bool operator==(const ClassTest& other) const {
    return field==other.field;
  }

  bool operator<(const ClassTest& other) const {
    return field<other.field;
  }
};

struct ClassTest2 : public Object {
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT)

  const int field;

protected:
  ClassTest2(const int field)
    : field(field) {}
public:

  bool operator==(const ClassTest2& other) const {
    return field==other.field;
  }
};

const int ClassTest::static_const;

GEODE_DEFINE_TYPE(ClassTest)
GEODE_DEFINE_TYPE(ClassTest2)

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

  {
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
      .GEODE_METHOD(dump_mem)
      .getattr()
      .setattr()
      .hash()
      .compare()
      ;
  } {
    typedef ClassTest2 Self;
    Class<Self>("ClassTest2")
      .GEODE_INIT(int)
      .compare()
      ;
  }
}
