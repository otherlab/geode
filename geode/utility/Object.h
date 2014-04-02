// Object: A base class for reference counted C++ objects.
#pragma once

// Since all instances of Object are reference counted, they can be owned only through Ref<T> and Ptr<T>,
// which manage the reference count.  To allocate a new object of type T, use new_<T>(...), which returns a Ref.
// Classes derived from Object are noncopyable by default.
//
// Since enable_shared_from_this adds overhead, we also declare a smaller base class Owner
// for use by Buffer and Array.

#include <geode/utility/config.h>
#include <geode/utility/forward.h>
#include <geode/utility/smart_ptr.h>
namespace geode {

using std::string;

// Lightweight version for use by Buffer.  Does not use enable_shared_from_this,
class GEODE_CORE_CLASS_EXPORT Owner {
public:
  GEODE_CORE_EXPORT virtual ~Owner();
};

class GEODE_CORE_CLASS_EXPORT Object : public Owner, public enable_shared_from_this<Object> {
public:
  GEODE_NEW_FRIEND
  typedef Object Base; // The hierarchy stops here

protected:
  GEODE_CORE_EXPORT Object();

  // Make noncopyable by default
  Object(const Object&) = delete;
  void operator=(const Object&) = delete;

  virtual string repr() const;
};

}
#include <geode/utility/Ref.h>
#include <geode/utility/Ptr.h>
#include <geode/utility/new.h>
