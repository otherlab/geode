//#####################################################################
// Class ArrayExpression
//#####################################################################
#pragma once

#include <geode/array/ArrayIter.h>
#include <geode/utility/type_traits.h>
#include <boost/static_assert.hpp>
namespace geode {

struct ConstantSizeArrayExpressionBase {};
template<int m_> struct ArrayExpressionBase : public ConstantSizeArrayExpressionBase { enum {m = m_}; };
template<> struct ArrayExpressionBase<-1> {};

template<class TA,class Enable=void> struct SizeIfConstant { enum {m = -1}; };
template<class T,int d> struct SizeIfConstant<Vector<T,d>> { enum {m = d}; };
template<class TA> struct SizeIfConstant<TA,typename enable_if<is_base_of<ConstantSizeArrayExpressionBase,TA>>::type> { enum {m = TA::m}; };

#ifdef GEODE_VARIADIC
template<int m,class... Args> struct SameSizeHelper;
template<int m_> struct SameSizeHelper<m_> { enum {m = m_}; };
template<int m_,class A,class... Args> struct SameSizeHelper<m_,A,Args...> {
  enum {Am = SizeIfConstant<typename remove_const_reference<A>::type>::m};
  BOOST_STATIC_ASSERT((m_<0 || Am<0 || m_==Am));
  enum {m = SameSizeHelper<m_<0?Am:m_,Args...>::m};
};
#else
template<int m_,class A0=void,class A1=void> struct SameSizeHelper {
  enum {m0 = SizeIfConstant<typename remove_const_reference<A0>::type>::m};
  enum {m1 = SizeIfConstant<typename remove_const_reference<A1>::type>::m};
  BOOST_STATIC_ASSERT((m0<0 || m1<0 || m0==m1));
  enum {m = m0>=0?m0:m1};
};
#endif

#ifdef GEODE_VARIADIC
template<class T_,class TArray,class... Args>
class ArrayExpression : public ArrayBase<T_,TArray>, public ArrayExpressionBase<SameSizeHelper<-1,Args...>::m>
#else
template<class T_,class TArray,class A0,class A1>
class ArrayExpression : public ArrayBase<T_,TArray>, public ArrayExpressionBase<SameSizeHelper<-1,A0,A1>::m>
#endif
{
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
