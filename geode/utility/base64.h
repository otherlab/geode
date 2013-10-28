// Base 64 encoding and decoding
#pragma once

#include <geode/array/RawArray.h>
#include <string>
namespace geode {

GEODE_CORE_EXPORT string base64_encode(const string& src);
GEODE_CORE_EXPORT string base64_encode(RawArray<const uint8_t> src);

GEODE_CORE_EXPORT string base64_decode(const string& src);

}
