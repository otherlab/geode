#include <geode/mesh/TriangleSoup.h>
#include <geode/array/Field.h>

namespace geode {

Tuple<Ref<const TriangleSoup>, Array<Vector<real,3>>> lower_hull(TriangleSoup const &imesh, Array<Vector<real,3>> const &pos, Vector<real,3> up, const real ground_offset);

}
