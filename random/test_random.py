#!/usr/bin/env python

from __future__ import division
from numpy import *
from other.core.random import *
from other.core import *
import hashlib
import numpy

def test_sobol():
  m,n = 128,128
  box = Box(0,(m,n,m))
  sobol = Sobol(box)
  print box

  im = tile(1,(m,n,3))

  count = m*n//100*10
  for _ in range(count):
    X = sobol.vector()
    assert box.thickened(.1).lazy_inside(X)
    for i in range(3):
      cell = list(map(int,X))
      del cell[i]
      im[cell[0],cell[1],i] = 0

  file = '/tmp/sobol.png'
  Image.write(file,im.astype(python.real))
  hash = hashlib.sha1(open(file).read()).hexdigest()
  assert hash=='1492d817fdb75a6de90bf174dbd05d222f42676d'

def test_threefry():
  # Known answer test vectors for 20 round threefry2x64 from the Random123 distribution
  kat = '''0000000000000001 0000000000000000   0000000000000001 0000000000000000   76f8c465410f1b27 d44c2d67df04a330
           0000000000000000 0000000000000001   0000000000000001 0000000000000000   a8049503dd3079f9 3d5ddeed522c0ede
           0000000000000000 ffffffffffffffff   0000000000000001 0000000000000000   5e53259aa4258a55 0e8f38c365945d25
           0000000000000000 8000000000000000   0000000000000001 0000000000000000   361a8be863ffb732 a4b1a2eeb58e74d0
           243f6a8885a308d3 13198a2e03707344   0000000000000001 0000000000000000   fd6ae8f34dc33d12 038c18af4795d22a
           a4093822299f31d0 082efa98ec4e6c89   0000000000000001 0000000000000000   7ac8bd11c1f70ff1 74639e2ca058255e
           452821e638d01377 be5466cf34e90c6c   0000000000000001 0000000000000000   ed08efd2a170bbd7 0dd10199b9784662
           0000000000000001 0000000000000000   0000000000000000 0000000000000001   ff2b78b5ab41d8da f62ebfe044d2eda8
           0000000000000000 0000000000000001   0000000000000000 0000000000000001   5f3a212f661cd020 986a3ba650b3fe67
           0000000000000000 ffffffffffffffff   0000000000000000 0000000000000001   6a6b93a7418044f4 87b8ba42ce2ff9bf
           0000000000000000 8000000000000000   0000000000000000 0000000000000001   bae5e0296c6a477d 0fbb8a9f8a4ce574
           243f6a8885a308d3 13198a2e03707344   0000000000000000 0000000000000001   99cec910eff5ec80 590c31d2679b536f
           a4093822299f31d0 082efa98ec4e6c89   0000000000000000 0000000000000001   02f8b55a501f2281 d76b4bf86e78c17a
           452821e638d01377 be5466cf34e90c6c   0000000000000000 0000000000000001   85ed5f68252b8dad 94e7c2f00c87c774
           0000000000000001 0000000000000000   0000000000000000 ffffffffffffffff   2315d2c0d1827ca7 6c9edc5ea9168247
           0000000000000000 0000000000000001   0000000000000000 ffffffffffffffff   e673b4093ab96a92 e6b9195814502ad9
           0000000000000000 ffffffffffffffff   0000000000000000 ffffffffffffffff   617e1767c6bab4b3 9defeccc66ef5483
           0000000000000000 8000000000000000   0000000000000000 ffffffffffffffff   64008223d83f51ea 703e62828caafe5c
           243f6a8885a308d3 13198a2e03707344   0000000000000000 ffffffffffffffff   55c5791f5e6a445c 6244e287ceff815b
           a4093822299f31d0 082efa98ec4e6c89   0000000000000000 ffffffffffffffff   69f5517074ffb421 364c6cb5114df09b
           452821e638d01377 be5466cf34e90c6c   0000000000000000 ffffffffffffffff   8abfad0acf126feb 7e4838608dff93a6
           0000000000000001 0000000000000000   0000000000000000 8000000000000000   f3113561b7808541 e1d6850fd01f03c8
           0000000000000000 0000000000000001   0000000000000000 8000000000000000   1f254bdb6686f3f7 2defd1bc9a4e7d58
           0000000000000000 ffffffffffffffff   0000000000000000 8000000000000000   becfaedd81933ce0 91105295132db554
           0000000000000000 8000000000000000   0000000000000000 8000000000000000   08e7eef4615bbae3 16c446fef3ff2fa6
           243f6a8885a308d3 13198a2e03707344   0000000000000000 8000000000000000   58af89abe6d07cab 12e1901ad854ba14
           a4093822299f31d0 082efa98ec4e6c89   0000000000000000 8000000000000000   469ded4657b380b5 4e18d924d1d191f4
           452821e638d01377 be5466cf34e90c6c   0000000000000000 8000000000000000   89983220cc01af30 756813c51f68660f'''
  kat = map(lambda s:int(s,16),kat.split())
  def combine(lo,hi):
    return (hi<<64)+lo
  for i in xrange(len(kat)//6):
    ctr = combine(kat[6*i+0],kat[6*i+1])
    key = combine(kat[6*i+2],kat[6*i+3])
    val = combine(kat[6*i+4],kat[6*i+5])
    assert threefry(key,ctr)==val

def test_bits():
  # Extremely weak consistency checks on bits.  These are to catch code bugs in Random, not errors in the
  # underlying generator.  test_threefry should suffice for the latter.
  import scipy.stats
  random = Random(5)
  n = 2**15
  mean = n/2
  var = n/4
  X = (hstack(random_bits_test(random,n))-mean)/sqrt(var)
  m = len(X)
  # See http://math.stackexchange.com/questions/89030/expectation-of-the-maximum-of-gaussian-random-variables
  gamma = 0.5772156649015328606065120900824024310421
  expected = sqrt(2*log(m)-log(4*pi*log(m)-2*pi*log(2*pi)))*(1+gamma/log(m))
  min,max = X.min(),X.max()
  print 'n = %d, m = %d, expected extreme = %g, error range = %g %g'%(n,m,expected,min,max)
  for e in -min,max:
    assert relative_error(expected,e)<.1

def test_distributions():
  import scipy.stats
  numpy.random.seed(7)
  random = Random(6)
  n = 2**20
  def test(name,dist,cuts,X):
    cuts = hstack([-inf,cuts,inf])
    observed = histogram(X,cuts)[0]
    cdf = dist.cdf(cuts)
    expected = n*(cdf[1:]-cdf[:-1])
    chi2,p = scipy.stats.chisquare(observed,expected)
    print '%s: chi^2 = %g, p = %g'%(name,chi2,p)
    assert p>.4
    # Check independence of adjacent entries
    observed = histogram2d(X[:-1],X[1:],cuts)[0]
    chi2,p,_,_ = scipy.stats.chi2_contingency(observed)
    print '%s independence: chi^2 = %g, p = %g'%(name,chi2,p)
    assert p>.4
  # Test normal
  test('normal',scipy.stats.norm,arange(-3,3.001,.03),random.normal(n))
  # Test uniform
  X = random.uniform(n)
  assert all(0<=X) and all(X<1)
  test('uniform',scipy.stats.uniform,arange(.005,1,.01),X)
  # Test uniform int
  for lo,hi in (0,7),(-4,4):
    X = random.uniform_int(lo,hi,n)
    assert X.dtype==int32 and all(lo<=X) and all(X<hi)
    test('int %d %d'%(lo,hi),scipy.stats.randint(lo,hi),arange(lo,hi-1)+.5,X)

def test_permute():
  # Note: This tests only that random_permute(n,_) is a valid permutation, not for pseudorandomness.
  numpy.random.seed(7810131)
  for n in 1025,14710134,2**32,2**32+1,2**64-1:
    key = threefry(18371,n)
    xs = numpy.fromstring(numpy.random.bytes(8*16),dtype=uint64)%n
    for x in xs:
      y = random_permute(n,key,x)
      assert x==random_unpermute(n,key,y)
      assert y<n
  # For small n, we check that all permutations are realized
  fac = 1
  for n in 1,2,3,4,5:
    fac *= n
    perms = set()
    for key in xrange(int(1+10*fac*log(fac))):
      perm = tuple(random_permute(n,key,i) for i in xrange(n))
      for i,pi in enumerate(perm):
        assert i==random_unpermute(n,key,pi)
      perms.add(perm)
    assert len(perms)==fac

if __name__=='__main__':
  test_permute()
  test_bits()
  test_distributions()
  test_sobol()
