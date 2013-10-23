#pragma once

#include <geode/utility/macro_map.h>
#include <geode/utility/remove_commas.h>

#define GEODE_USING_HELPER(name) using Base::name;
#define GEODE_USING(...) GEODE_REMOVE_COMMAS(GEODE_MAP(GEODE_USING_HELPER,__VA_ARGS__))
