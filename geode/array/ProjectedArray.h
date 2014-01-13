//#####################################################################
// Class ProjectedArray
//#####################################################################
#pragma once

#include <geode/array/forward.h>
#include <geode/vector/forward.h>
#include <geode/utility/HasCheapCopy.h>
#include <geode/utility/type_traits.h>
#include <boost/mpl/assert.hpp>
namespace geode {

template<class TArray,class TProjector> struct IsArray<ProjectedArray<TArray,TProjector> >:public mpl::true_{};
template<class TArray,class TProjector> struct HasCheapCopy<ProjectedArray<TArray,TProjector> >:public mpl::true_{};

template<class TArray,class TProjector> struct ProjectedArrayElement {
  typedef typename remove_const_reference<decltype(declval<TProjector>()(declval<TArray>()[0]))>::type type;
};

template<class TArray,class TProjector>
class ProjectedArray : public ArrayExpression<typename ProjectedArrayElement<TArray,TProjector>::type,ProjectedArray<TArray,TProjector>,TArray>
                     , private TProjector {
  typedef typename ProjectedArrayElement<TArray,TProjector>::type T;
  typedef typename mpl::if_<HasCheapCopy<TArray>,const TArray,const TArray&>::type TArrayView;
public:
  typedef T Element;

  TArrayView array;

  ProjectedArray(TArrayView array)
    : array(array) {
    BOOST_MPL_ASSERT((is_empty<TProjector>));
  }

  ProjectedArray(TArrayView array, const TProjector& projector)
    : TProjector(projector), array(array) {}

  ProjectedArray(const ProjectedArray<typename remove_const<TArray>::type,TProjector>& projected_array)
    : TProjector(projected_array.projector()), array(projected_array.array) {}

  const TProjector& projector() const {
    return *this;
  }

  int size() const {
    return array.size();
  }

  auto operator[](const int i) const
    -> decltype(declval<TProjector>()(array[i])) {
    return TProjector::operator()(array[i]);
  }

  ProjectedArray operator=(const ProjectedArray& source) {
    return ArrayBase<T,ProjectedArray>::operator=(source);
  }

  template<class TArray2> ProjectedArray operator=(const TArray2& source) {
    return ArrayBase<T,ProjectedArray>::operator=(source);
  }
};

template<class TStruct,class TField,TField TStruct::* field>
struct FieldProjector {
  const TField& operator()(const TStruct& element) const {
    return element.*field;
  }

  TField& operator()(TStruct& element) const {
    return element.*field;
  }
};

struct IndexProjector {
  const int index;

  IndexProjector(const int index)
    : index(index) {}

  template<class TArray> auto operator()(TArray& array) const
    -> decltype(array[index]) {
    return array[index];
  }
};

struct PointerProjector { // turns shared_ptr<T>, etc., into T*
  template<class TP> auto operator()(const TP& pointer) const
    -> decltype(&*pointer) {
    return &*pointer;
  }
};

}
#include <geode/array/ArrayBase.h>
namespace geode {

template<class TArray> static inline ProjectedArray<const TArray,PointerProjector> raw_pointers(const TArray& array) {
  return ProjectedArray<const TArray,PointerProjector>(array);
}

}
