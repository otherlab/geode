set(module_SOURCES
  AirPressure.cpp
  AxisPins.cpp
  BindingSprings.cpp
  ConstitutiveModel.cpp
  CubicHinges.cpp
  DiagonalizedIsotropicStressDerivative.cpp
  EtherDrag.cpp
  FiniteVolume.cpp
  Force.cpp
  Gravity.cpp
  LinearBendingElements.cpp
  LinearFiniteVolume.cpp
  LinearFiniteVolumeHex.cpp
  NeoHookean.cpp
  ParticleBindingSprings.cpp
  Pins.cpp
  PlasticityModel.cpp
  RotatedLinear.cpp
  SimpleShell.cpp
  Springs.cpp
  StrainMeasure.cpp
  StrainMeasureHex.cpp
  SurfacePins.cpp
)

set(module_HEADERS
  AirPressure.h
  AnisotropicConstitutiveModel.h
  AxisPins.h
  BindingSprings.h
  ConstitutiveModel.h
  CubicHinges.h
  DiagonalizedIsotropicStressDerivative.h
  DiagonalizedStressDerivative.h
  EtherDrag.h
  FiniteVolume.h
  Force.h
  forward.h
  Gravity.h
  IsotropicConstitutiveModel.h
  LinearBendingElements.h
  LinearFiniteVolume.h
  LinearFiniteVolumeHex.h
  ParticleBindingSprings.h
  Pins.h
  PlasticityModel.h
  SimpleShell.h
  Springs.h
  StrainMeasure.h
  StrainMeasureHex.h
  SurfacePins.h
)

install_geode_headers(force ${module_HEADERS})

add_geode_module(force ${module_SOURCES})
