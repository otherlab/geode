// Base 64 encoding and decoding
#pragma once

#include <other/core/array/RawArray.h>
#include <string>
namespace other {

string base64_encode(const string& src) OTHER_CORE_EXPORT;
string base64_encode(RawArray<const uint8_t> src) OTHER_CORE_EXPORT;

string base64_decode(const string& src) OTHER_CORE_EXPORT;

}
