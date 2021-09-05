//#####################################################################
// Module openmesh
//#####################################################################
#include <geode/python/wrap.h>
#include <geode/openmesh/TriMesh.h>

using namespace geode;

static bool openmesh_enabled() {
#ifdef GEODE_OPENMESH
  return true;
#else
  return false;
#endif
}

static std::string openmesh_qt_read_filters() {
#ifdef GEODE_OPENMESH
  return OpenMesh::IO::IOManager().qt_read_filters();
#else
  return "";
#endif
}
static std::string openmesh_qt_write_filters() {
#ifdef GEODE_OPENMESH
  return OpenMesh::IO::IOManager().qt_write_filters();
#else
  return "";
#endif
}

// Static builds of OpenMesh define 'instance' variables for mesh readers and writers
// Although not directly used, their initialization indirectly constructs readers and writers for each format
// Since we can't retroactively apply the unused attribute we 'use' the variables here
#if defined(GEODE_OPENMESH) && defined(OM_STATIC_BUILD) && ((OM_STATIC_BUILD + 0) != 0)
#define MAKE_DUMMY(ext) \
  GEODE_UNUSED static OpenMesh::IO::BaseReader* dummy_reference_for_ ## ext ## reader = OpenMesh::IO::ext ## ReaderInstance; \
  GEODE_UNUSED static OpenMesh::IO::BaseWriter* dummy_reference_for_ ## ext ## writer = OpenMesh::IO::ext ## WriterInstance;
MAKE_DUMMY(OFF)
MAKE_DUMMY(OBJ)
MAKE_DUMMY(PLY)
MAKE_DUMMY(STL)
MAKE_DUMMY(OM)
#endif

void wrap_openmesh() {
  GEODE_FUNCTION(openmesh_enabled)
  GEODE_FUNCTION(openmesh_qt_read_filters)
  GEODE_FUNCTION(openmesh_qt_write_filters)
#ifdef GEODE_OPENMESH
  GEODE_WRAP(trimesh)
  GEODE_WRAP(openmesh_decimate)
  GEODE_WRAP(curvature)
    //  GEODE_WRAP(smooth)
#endif
}
