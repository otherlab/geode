from other.core.math import *
from numpy import *

def test_uint128():
  try:
    uint128_test(1<<128,0)
  except OverflowError:
    pass
  try:
    uint128_test(-1,0)
  except OverflowError:
    pass
  m64 = (1<<64)-1
  m128 = (1<<128)-1
  for i in xrange(100):
    x = threefry(0,i)
    y = threefry(1,i)
    r = uint128_test(x,y)
    assert r[0]==(-7)&m64
    assert r[1]==(-7)&m128
    assert r[2]==x
    assert r[3]==(x+y)&m128
    assert r[4]==(x-y)&m128
    assert r[5]==(x*y)&m128
    assert r[6]==(x<<5)&m128
    assert r[7]==x>>7
  for x in 0,19731,74599465401539225308381206871:
    assert uint128_str_test(x)=='0x%x'%x
