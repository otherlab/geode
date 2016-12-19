set(module_SOURCES
  Array2d.cpp
  Array.cpp
  NdArray.cpp
  Nested.cpp
  permute.cpp
  RawArray.cpp
  stencil.cpp
)

set(module_HEADERS
  alloca.h
  amap.h
  arange.h
  Array2d.h
  Array3d.h
  Array4d.h
  ArrayAbs.h
  ArrayBase.h
  ArrayDifference.h
  ArrayExpression.h
  Array.h
  ArrayIter.h
  ArrayLeftMultiple.h
  ArrayNdBase.h
  ArrayNegation.h
  ArrayPlusScalar.h
  ArrayProduct.h
  ArraySum.h
  ConstantMap.h
  convert.h
  Field.h
  forward.h
  IndirectArray.h
  NdArray.h
  NestedField.h
  Nested.h
  permute.h
  ProjectedArray.h
  RawArray.h
  RawField.h
  RawStack.h
  reversed.h
  SmallArray.h
  sort.h
  stencil.h
  Subarray.h
  UntypedArray.h
  view.h
)

install_geode_headers(array ${module_HEADERS})

add_geode_module(array ${module_SOURCES})
