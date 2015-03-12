lib_setup_args="sse=0 shared=0 use_python=0 use_libjpeg=0 use_libpng=0 use_openexr=0 use_boost=0 use_openmesh=1 use_gmp=1"
echo "Using: $lib_setup_args"
if [ "$1" = "clean" ] || [ "$1" = "distclean" ]; then
	scons -c --config=force $lib_setup_args
else
  for type in debug release; do
  	# arch='nocona'
    scons -j7 prefix='#build/$arch/$type' arch='nocona' install=0 $lib_setup_args \
      openmesh_libpath='' openmesh_publiclibs='' openmesh_include='#/../OpenMesh-2.0/src' openmesh_linkflags='../OpenMesh-2.0/build/Build/lib/OpenMesh/libOpenMeshCore.dylib ../OpenMesh-2.0/build/Build/lib/OpenMesh/libOpenMeshTools.dylib' \
      gmp_linkflags='../gmp/.libs/libgmp.dylib' gmp_include='#/../gmp' gmp_libpath='' gmp_publiclibs='' \
      type=$type cxxflags_extra='-std=c++11 -stdlib=libc++' linkflags_extra='-stdlib=libc++'
  done
fi
