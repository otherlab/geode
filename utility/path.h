// Path convenience functions
#pragma once

// We could use boost::filesystem for this, but it's nice to
// keep core independent of as much as possible.

#include <other/core/utility/config.h>
#include <string>
namespace other {
namespace path {

using std::string;

string extension(const string& path) OTHER_EXPORT;

}
}
