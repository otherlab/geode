Import('env library external')

external('openmesh',libpath=['/usr/local/lib/OpenMesh'],libs=['OpenMeshCore','OpenMeshTools'],requires=['boost_link'])

env = env.Clone(use_libpng=1,use_libjpeg=1,use_openexr=1,use_openmesh=1)
library(env,'other_core')
