Import('env library external')

external('openmesh',libpath=['/usr/local/lib/OpenMesh'],flags=['USE_OPENMESH'],libs=['OpenMeshCore','OpenMeshTools'],requires=['boost_link'])

env = env.Clone(use_libpng=1,use_libjpeg=1,use_openexr=0,use_openmesh=1)
# Minimal dependencies:
# env = env.Clone(use_libpng=0,use_libjpeg=0,use_openexr=0,use_openmesh=0)
library(env,'other_core')
