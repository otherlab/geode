Import('env library external windows clang toplevel')

external('openmesh',libpath=['/usr/local/lib/OpenMesh'],flags=['USE_OPENMESH'],libs=['OpenMeshCore','OpenMeshTools'],requires=['boost_link'])

external('gmm',flags=['_SCL_SECURE_NO_DEPRECATE'] if windows else [],cxxflags=' -Wno-return-type-c-linkage' if clang else '')
toplevel('gmm','#data/gmm-3.0/gmm')

env = env.Clone(use_libpng=1,use_libjpeg=1,use_openexr=0,use_openmesh=1,use_gmm=1)
# Minimal dependencies:
# env = env.Clone(use_libpng=0,use_libjpeg=0,use_openexr=0,use_openmesh=0,use_python=0)
library(env,'other_core')
