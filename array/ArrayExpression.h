//#####################################################################
// Class ArrayExpression
//#####################################################################
#pragma once

#include <other/core/array/ArrayIter.h>
#include <other/core/utility/remove_const_reference.h>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/static_assert.hpp>
namespace other {

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
  typedef ArrayIter<TArray> iterator;
  typedef ArrayIter<const TArray> const_iterator;

  ArrayIter<TArray> begin() const {
    return ArrayIter<TArray>(derived(),0);}

  ArrayIter<TArray> end() const {
    return ArrayIter<TArray>(derived(),derived().size());
  }
};

}
