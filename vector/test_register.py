from other.core.vector import *
from numpy import *

def test_register():
    random.seed(217301)
    for _ in xrange(10):
        for d in 2,3:
            X0 = random.randn(10,d)
            t = random.randn(d)
            r,_ = linalg.qr(random.randn(d,d))
            if linalg.det(r)<0:
                r[0] = -r[0]
            X1 = t + Matrix(r)*X0
            f = rigid_register(X0,X1)
            assert relative_error(t,f.t) < 1e-5
            assert relative_error(r,f.r.matrix()) < 1e-5
