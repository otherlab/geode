set(module_SRCS
  Heap.cpp
  RankedTree.cpp
  Tuple.cpp
)

set(module_HEADERS
  Empty.h
  forward.h
  Hashtable.h
  Heap.h
  OperationHash.h
  Pair.h
  Quad.h
  Queue.h
  Quintuple.h
  RankedTree.h
  RankedTree_p.h
  Singleton.h
  Stack.h
  Triple.h
  Tuple.h
  UnionFind.h
)

install_geode_headers(structure ${module_HEADERS})

add_geode_module(structure ${module_SRCS})
