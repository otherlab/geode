#pragma once

#include <other/core/utility/macro_map.h>
#include <other/core/utility/remove_commas.h>

#define OTHER_USING_HELPER(name) using Base::name;
#define OTHER_USING(...) OTHER_REMOVE_COMMAS(OTHER_MAP(OTHER_USING_HELPER,__VA_ARGS__))
