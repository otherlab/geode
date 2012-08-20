//#####################################################################
// Class ArrayExpression
//#####################################################################
#pragma once

#include <other/core/array/forward.h>
#include <other/core/utility/remove_const_reference.h>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/utility/declval.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/static_assert.hpp>
#include <iterator>
namespace other {

template<class TArray> class ArrayStlIterator {
  TArray* array;
  int index;
public:
  typedef std::random_access_iterator_tag iterator_category;
  typedef decltype(boost::declval<TArray>()[0]) reference;
  typedef typename boost::remove_reference<reference>::type value_type;
  typedef int difference_type;
  typedef value_type* pointer;

  ArrayStlIterator()
    : array(0), index(0) {}

  ArrayStlIterator(TArray& array, const int index)
    : array(&array), index(index) {}

  void operator++() {
    index++;
  }

  bool operator==(const ArrayStlIterator& other) {
    return index==other.index; // assume array==other.array
  }

  bool operator!=(const ArrayStlIterator& other) {
    return index!=other.index; // assume array==other.array
  }

  reference operator*() const {
    return (*array)[index];
  }
};

struct ConstantSizeArrayExpressionBase {};
template<int m_> struct ArrayExpressionBase : public ConstantSizeArrayExpressionBase { enum {m = m_}; };
template<> struct ArrayExpressionBase<-1> {};

template<class TA,class Enable=void> struct SizeIfConstant { enum {m = -1}; };
template<class T,int d> struct SizeIfConstant<Vector<T,d>> { enum {m = d}; };
template<class TA> struct SizeIfConstant<TA,typename boost::enable_if<boost::is_base_of<ConstantSizeArrayExpressionBase,TA>>::type> { enum {m = TA::m}; };

template<int m,class... Args> struct SameSizeHelper;
template<int m_> struct SameSizeHelper<m_> { enum {m = m_}; };
template<int m_,class A,class... Args> struct SameSizeHelper<m_,A,Args...> {
  enum {Am = SizeIfConstant<typename remove_const_reference<A>::type>::m};
  BOOST_STATIC_ASSERT((m_<0 || Am<0 || m_==Am));
  enum {m = SameSizeHelper<m_<0?Am:m_,Args...>::m};
};

template<class T_,class TArray,class... Args>
class ArrayExpression : public ArrayBase<T_,TArray>, public ArrayExpressionBase<SameSizeHelper<-1,Args...>::m> {
  using ArrayBase<T_,TArray>::derived;
public:
  typedef ArrayStlIterator<TArray> iterator;
  typedef ArrayStlIterator<const TArray> const_iterator;

  ArrayStlIterator<TArray> begin() {
    return ArrayStlIterator<TArray>(derived(),0);}

  ArrayStlIterator<const TArray> begin() const {
    return ArrayStlIterator<const TArray>(derived(),0);
  }

  ArrayStlIterator<TArray> end() {
    return ArrayStlIterator<TArray>(derived(),derived().size());
  }

  ArrayStlIterator<const TArray> end() const {
    return ArrayStlIterator<const TArray>(derived(),derived().size());
  }
};
}
