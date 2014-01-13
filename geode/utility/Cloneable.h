//#####################################################################
// Class Cloneable
//#####################################################################
#pragma once

#include <geode/utility/debug.h>
#include <geode/utility/type_traits.h>
#include <boost/mpl/and.hpp>
#include <boost/mpl/assert.hpp>
#include <typeinfo>
namespace geode {

class CloneableBase; // Cloneable abstract base classes inherit from this directly
template<class T> class CloneArray;

template<class T> struct IsCloneable : public boost::mpl::and_<
  is_base_of<CloneableBase,T>, // must eventually derive from CloneableBase
  is_convertible<T*,CloneableBase*> >{}; // ensure derivation isn't ambiguous

class CloneableBase {
public:
  virtual ~CloneableBase() {}

  CloneableBase* clone() const {
    return clone_implementation();
  }

  CloneableBase* clone_default() const { // creates a default constructed copy of the same type
    return clone_default_implementation();
  }

protected:
  template<class T> friend class CloneArray;

  virtual CloneableBase* clone_implementation() const = 0;
  virtual CloneableBase* clone_default_implementation() const = 0;
  virtual size_t sizeof_clone() const = 0;
  virtual CloneableBase* placement_clone(void* memory) const = 0;
};

template<class TDerived,class TBase=CloneableBase>
class CloneableAbstract : public TBase {
  BOOST_MPL_ASSERT((IsCloneable<TBase>));
  using TBase::clone_implementation;using TBase::clone_default_implementation;
  template<class T> friend class CloneArray;
public:

  TDerived* clone() const {
    return static_cast<TDerived*>(clone_implementation());
  }

  TDerived* clone_default() const { // Create a default constructed copy of the same type
    return static_cast<TDerived*>(clone_default_implementation());
  }
};

template<class TDerived,class TBase=CloneableBase>
class Cloneable : public TBase {
  BOOST_MPL_ASSERT((IsCloneable<TBase>));
public:

  TDerived* clone() const {
    return static_cast<TDerived*>(clone_implementation());
  }

  TDerived* clone_default() const { // creates a default constructed copy of the same type
    return static_cast<TDerived*>(clone_default_implementation());
  }

private:
  template<class T> friend class CloneArray;

  virtual CloneableBase* clone_implementation() const {
    GEODE_ASSERT(typeid(*this)==typeid(TDerived)); // avoid slicing errors
    TDerived* clone = new TDerived();
    clone->clone_helper(static_cast<const TDerived&>(*this));
    return clone;
  }

  virtual CloneableBase* clone_default_implementation() const {
    GEODE_ASSERT(typeid(*this)==typeid(TDerived)); // avoid slicing errors
    return new TDerived();
  }

  virtual size_t sizeof_clone() const {
    GEODE_ASSERT(typeid(*this)==typeid(TDerived)); // avoid horrible memory corruption errors
    return sizeof(TDerived);
  }

  virtual CloneableBase* placement_clone(void* memory) const {
    GEODE_ASSERT(typeid(*this)==typeid(TDerived)); // avoid slicing errors
    TDerived* clone = new(memory) TDerived();
    clone->clone_helper(static_cast<const TDerived&>(*this));
    return clone;
  }
};

}
