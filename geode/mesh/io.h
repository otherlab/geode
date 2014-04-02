// Mesh file I/O
#pragma once

#include <geode/mesh/TriangleSoup.h>
#include <geode/mesh/TriangleTopology.h>
namespace geode {

// Read a mesh format as triangle or polygon soup
GEODE_EXPORT Tuple<Ref<const TriangleSoup>,Array<Vector<real,3>>> read_soup(const string& filename);
GEODE_EXPORT Tuple<Ref<const PolygonSoup>,Array<Vector<real,3>>> read_polygon_soup(const string& filename);

// Read a mesh format and convert to a manifold mesh.  If the mesh is not manifold, an exception is thrown.
GEODE_EXPORT Tuple<Ref<const TriangleTopology>,Array<Vector<real,3>>> read_mesh(const string& filename);

// Write a mesh to a file
GEODE_EXPORT void write_mesh(const string& filename, const TriangleSoup& soup, RawArray<const Vector<real,3>> X);
GEODE_EXPORT void write_mesh(const string& filename, const PolygonSoup& soup, RawArray<const Vector<real,3>> X);
GEODE_EXPORT void write_mesh(const string& filename, const TriangleTopology& mesh, RawArray<const Vector<real,3>> X);

// write a mesh to a file, assuming the mesh position are stored with the default
// id and have type Vector<real,3>. 
GEODE_EXPORT void write_mesh(const string &filename, const MutableTriangleTopology &mesh);

}
