from geode import *
from numpy import *

def test_sse():
  random.seed(3871)
  for i in xrange(10):
    x = random.randn(2)
    assert all(x==sse_pack_unpack(x))
