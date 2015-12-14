#include <geode/exact/config.h>

////////////////////////////////////////////////////////////////////////////////
// Forward declarations for circle data types
namespace geode {
typedef Perturbation Pb;

template<Pb PS> struct ExactCircle;
template<Pb PS> struct ExactHorizontal;
template<Pb PS> struct ExactArc;
template<Pb PS> struct ExactHorizontalArc;

template<Pb PS> struct IncidentCircle;
template<Pb PS> struct IncidentHorizontal;

template<Pb PS> struct CircleIntersectionKey;
template<Pb PS> struct CircleIntersection;
template<Pb PS> struct HorizontalIntersection;

struct ApproxIntersection;

// Shorthand typedefs
typedef ExactCircle<Pb::Implicit> ExactCircleIm;
typedef ExactCircle<Pb::Explicit> ExactCircleEx;
} // namespace geode
