//#####################################################################
// Function hash
//#####################################################################
#pragma once

#include <other/core/utility/config.h>
#include <boost/static_assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/utility/enable_if.hpp>
#include <stdint.h>
#include <cstring>
#include <string>
namespace other {

struct Hash;
using std::string;
namespace mpl = boost::mpl;
BOOST_STATIC_ASSERT(sizeof(int)==4);

template<class T> struct is_packed_pod:public mpl::false_{};
template<class T> struct is_packed_pod<const T>:public is_packed_pod<T>{};
template<> struct is_packed_pod<bool>:public mpl::true_{};
template<> struct is_packed_pod<char>:public mpl::true_{};
template<> struct is_packed_pod<unsigned char>:public mpl::true_{};
template<> struct is_packed_pod<short>:public mpl::true_{};
template<> struct is_packed_pod<unsigned short>:public mpl::true_{};
template<> struct is_packed_pod<int>:public mpl::true_{};
template<> struct is_packed_pod<unsigned int>:public mpl::true_{};
template<> struct is_packed_pod<long>:public mpl::true_{};
template<> struct is_packed_pod<unsigned long>:public mpl::true_{};
template<> struct is_packed_pod<long long>:public mpl::true_{};
template<> struct is_packed_pod<unsigned long long>:public mpl::true_{};
template<> struct is_packed_pod<float>:public mpl::true_{};
template<> struct is_packed_pod<double>:public mpl::true_{};
template<class T> struct is_packed_pod<T*>:public mpl::true_{};

Hash hash_reduce(const char* key) OTHER_CORE_EXPORT;
Hash hash_reduce(const string& key) OTHER_CORE_EXPORT;
template<class T> inline typename boost::enable_if_c<is_packed_pod<T>::value && sizeof(T)<=4,int>::type hash_reduce(const T& key);
template<class T> inline typename boost::enable_if_c<(is_packed_pod<T>::value && sizeof(T)>4 && sizeof(T)<=8),Hash>::type hash_reduce(const T& key);
template<class T> inline typename boost::enable_if_c<(is_packed_pod<T>::value && sizeof(T)>8),Hash>::type hash_reduce(const T& key);

struct Hash {
  int val;

  explicit Hash()
    :val(32138912) {}

  template<class T0> Hash(const T0& k0)
    :val(mix(k0)) {}

  template<class T0,class T1> Hash(const T0& k0,const T1& k1)
    :val(mix(value(k0),value(k1))) {}

  template<class T0,class T1,class T2> Hash(const T0& k0,const T1& k1,const T2& k2)
    :val(mix(value(k0),value(k1),value(k2))) {}

  template<class T0,class T1,class T2,class T3> Hash(const T0& k0,const T1& k1,const T2& k2,const T3& k3)
    :val(mix(mix(value(k0),value(k1),value(k2)),value(k3))) {}

  template<class T0,class T1,class T2,class T3,class T4> Hash(const T0& k0,const T1& k1,const T2& k2,const T3& k3,const T4& k4)
    :val(mix(mix(value(k0),value(k1),value(k2)),value(k3),value(k4))) {}

  template<class T0,class T1,class T2,class T3,class T4,class T5> Hash(const T0& k0,const T1& k1,const T2& k2,const T3& k3,const T4& k4,const T5& k5)
    :val(mix(mix(mix(value(k0),value(k1),value(k2)),value(k3),value(k4)),value(k5))) {}

private:
  static int value(const int key) {
    return key;
  }

  static int value(const Hash key) {
    return key.val;
  }

  static int value(const uint64_t key) {
    return mix(key);
  }

  template<class T> static int value(const T& key) {
    return value(hash_reduce(key));
  }

  static int mix(unsigned key) {
    key += ~(key << 15);
    key ^=  (key >> 10);
    key +=  (key << 3);
    key ^=  (key >> 6);
    key += ~(key << 11);
    key ^=  (key >> 16);
    return (int)key;
  }

  static int mix(int key) {
    return mix(unsigned(key));
  }

  static int mix(uint64_t key) {
    key = (~key) + (key << 18);
    key = key ^ (key >> 31);
    key = key * 21;
    key = key ^ (key >> 11);
    key = key + (key << 6);
    key = key ^ (key >> 22);
    return (int)key;
  }

  static int mix(unsigned a,unsigned b) {
    return mix(uint64_t(a)<<32|b);
  }

  static int mix(unsigned a,unsigned b,unsigned c) {
    a-=b;a-=c;a^=(c>>13);
    b-=c;b-=a;b^=(a<<8);
    c-=a;c-=b;c^=(b>>13);
    a-=b;a-=c;a^=(c>>12);
    b-=c;b-=a;b^=(a<<16);
    c-=a;c-=b;c^=(b>>5);
    a-=b;a-=c;a^=(c>>3);
    b-=c;b-=a;b^=(a<<10);
    c-=a;c-=b;c^=(b>>15);
    return (int)c;
  }
};

template<class TArray> Hash hash_array(const TArray& array) {
  int size = array.size();
  if(!size) return Hash();
  if(size==1) return Hash(hash_reduce(array[0]));
  Hash h = size&1?Hash(array[0],array[1],array[2]):Hash(array[0],array[1]);
  for(int i=2+(size&1);i<size;i+=2) h = Hash(h,array[i],array[i+1]);
  return h;
}

template<class T> inline typename boost::enable_if_c<is_packed_pod<T>::value && sizeof(T)<=4,int>::type hash_reduce(const T& key) {
  int data = 0;
  memcpy(&data,&key,sizeof(key));
  return data;
}

template<class T> inline typename boost::enable_if_c<(is_packed_pod<T>::value && sizeof(T)>4 && sizeof(T)<=8),Hash>::type hash_reduce(const T& key) {
  uint64_t data = 0;
  memcpy(&data,&key,sizeof(key));
  return Hash(data);
}

template<class T> inline typename boost::enable_if_c<(is_packed_pod<T>::value && sizeof(T)>8),Hash>::type hash_reduce(const T& key) {
  const int n = (sizeof(T)+3)/4;
  int data[n];
  data[n-1] = 0;
  memcpy(data,&key,sizeof(key));
  Hash h = n&1?Hash(data[0],data[1],data[2]):Hash(data[0],data[1]);
  for(int i=2+(n&1);i<n;i+=2) h = Hash(h,data[i],data[i+1]);
  return h;
}

template<class T>
inline int hash(const T& key) {
  return Hash(hash_reduce(key)).val;
}

}
