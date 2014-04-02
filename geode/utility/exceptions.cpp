// C++ equivalents of Python exceptions

#include <geode/utility/exceptions.h>
#include <geode/utility/unordered.h>
namespace geode {

using std::endl;

#define EXCEPTIONS() \
  F(IOError) \
  F(OSError) \
  F(LookupError) \
  F(IndexError) \
  F(KeyError) \
  F(TypeError) \
  F(ValueError) \
  F(NotImplementedError) \
  F(AssertionError) \
  F(AttributeError) \
  F(ArithmeticError) \
  F(OverflowError) \
  F(ZeroDivisionError) \
  F(ReferenceError) \
  F(ImportError)

namespace {
template<class Error> class SavedRuntimeError : public SavedException {
public:
  const string what;

  SavedRuntimeError(const char* what)
    : what(what) {}

  void print(ostream& output, const string& where) const {
    output << where << ": " << typeid(Error).name() << ", " << what << endl;
  }

  void throw_() const {
    throw Error(what);
  }
};
}

template<class Error> static Ref<const SavedException> factory(const exception& error) {
  return new_<SavedRuntimeError<Error>>(error.what());
}

typedef function<Ref<const SavedException>(const exception&)> Factory;
typedef unordered_map<const type_info*,Factory> Factories;

// Start off with a hard coded set of understood exceptions
static Factories make_factories() {
  Factories factories;
  #define F(Error) factories[&typeid(Error)] = factory<Error>;
  EXCEPTIONS()
  #undef F
  return factories;
}

static Factories factories = make_factories();

void register_save(const type_info& type, const Factory& save) {
  factories[&type] = save;
}

Ref<const SavedException> save(const exception& error) {
  const auto it = factories.find(&typeid(error));
  if (it != factories.end())
    return it->second(error);
  else
    return new_<SavedRuntimeError<RuntimeError>>(error.what());
}

// Instantiate constructors and destructors
#define F(Error) \
  Error::Error(const string& message) : Base(message) {} \
  Error::~Error() throw () {}
EXCEPTIONS()
#undef F

}
