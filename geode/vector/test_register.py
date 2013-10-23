from geode.vector import *
from numpy import *

def test_register():
  random.seed(217301)
  for _ in xrange(10):
    for d in 2,3:
      X0 = random.randn(10,d)
      t = random.randn(d)
      A = random.randn(d,d)
      r,_ = linalg.qr(A)
      if linalg.det(r)<0:
          r[0] = -r[0]
      X1 = t + Matrix(r)*X0
      X2 = t + Matrix(A)*X0
      f = rigid_register(X0,X1)
      B = affine_register(X0,X2)
      assert relative_error(t,f.t) < 1e-5
      assert relative_error(r,f.r.matrix()) < 1e-5
      assert relative_error(t,B[:d,d]) < 1e-5
      assert relative_error(A,B[:d,:d]) < 1e-5
      assert all(B[d]==hstack([zeros(d),1]))
