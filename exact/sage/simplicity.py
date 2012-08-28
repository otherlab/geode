#!/usr/bin/env sage
# Simulation of simplicity analysis and code generation

'''
Copyright (C) 2012, Otherlab.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

# See http://arxiv.org/abs/math/9410209 for the original paper by Edelsbrunner and Mucke.
# See http://www.sagemath.org/doc/tutorial/tour_polynomial.html for a brief intro to polynomial math in Sage.

from __future__ import with_statement
from sage.all import *
from collections import defaultdict
from contextlib import contextmanager
import numpy
import inspect
import time
import copy

### Logging

indent = [0]

@contextmanager
def scope(name):
  start = time.time()
  print '  '*indent[0]+name
  indent[0] += 1
  try:
    yield
  finally:
    indent[0] -= 1
    print '%*sEND %-*s%8.4f s'%(2*indent[0],'',60-2*indent[0]-4,name,time.time()-start)

def info(s):
  print '  '*indent[0]+s

### Utilities

def subset(a,p):
  return [a[i] for i in p]

def all_subsets(p,k):
  n = len(p)
  def helper(k,start):
    if not k:
      yield ()
    else:
      for i in xrange(start,n-k+1):
        for r in helper(k-1,i+1):
          yield (p[i],)+r
  return helper(k,0)

def all_permutations(p):
  n = len(p)
  def helper(k,p):
    if k==n:
      yield p
    for i in xrange(k,n):
      q = p[:]
      q[k],q[i] = q[i],q[k]
      for r in helper(k+1,q):
        yield r
  return map(tuple,helper(0,p))

def permutation_id(x):
  # Map a permutation to a unique integer id in range(n!).
  # This must match the C++ version in permutation_id.h.
  x = list(x)
  n = len(x)
  id = 0
  for i in xrange(n-1):
    j = i+numpy.argmin(x[i:])
    x[i],x[j] = x[j],x[i]
    id = (n-i)*id+j-i
  return id

def smallest_unsigned_integer(bound):
  for bits in 8,16:
    if bound<2**bits:
      return 'uint%d_t'%bits
  assert 0

def integer_table_size(table):
  assert min(table)>=0
  TI = smallest_unsigned_integer(max(table)+1)
  return int(TI[3:-2])//8*len(table) # This isn't one of my finest moments

def integer_table(name,table):
  TI = smallest_unsigned_integer(max(table)+1)
  return 'static const %s %s[%d] = {%s};'%(TI,name,len(table),','.join(map(str,table)))

### Basic linear algebra

def dot(u,v):
  return u.dot_product(v)

def cross(u,v):
  if len(u)==len(v)==2:
    return u[0]*v[1]-u[1]*v[0]
  return u.cross_product(v)

def triple(u,v,w):
  a,b,c = m
  return u[0]*cross(v[1:],w[1:])+v[0]*cross(w[1:],u[1:])+w[0]*cross(u[1:],v[1:])

def det(*m):
  return (cross if len(m)==2 else triple)(*m)

### Code generation

class Values(object):
  def __init__(self,ring):
    self.ring = ring
    self.d = {}

  def __len__(self):
    return len(self.d)

  def __getitem__(self,k):
    return self.d[self.ring(k)]

  def __setitem__(self,k,v):
    self.d[self.ring(k)] = v

  def __contains__(self,k):
    return self.ring(k) in self.d

  def values(self):
    return self.d.values()

class Block(object):
  def __init__(self,ring,vars,mul,tmp='v'):
    self.vars = vars
    self.values = Values(ring) # Map from expression to (op,args)
    for v,s in vars.iteritems():
      self.values[v] = s,()
    self.tmp = tmp
    self.next = 0
    self.mul = mul

  def ensure(self,e):
    '''Ensure that either e is available'''
    values = self.values
    if e in values or e.is_numeric():
      return
    u,v = SR.wild(0),SR.wild(1)
    m = e.match(-u)
    if m:
      u = m[u]
      self.ensure(u)
      values[e] = '-',(u,)
      return
    for op,exp in ('-',u-v),('+',u+v),('*',u*v):
      m = e.match(exp)
      if m:
        u,v = m[u],m[v]
        self.ensure(u)
        self.ensure(v)
        values[e] = op,(u,v)
        return
    m = e.match(u**v)
    if m:
      u = m[u]
      v = m[v]
      op = {2:'sqr',3:'cube'}[v]
      self.ensure(u)
      values[e] = op,(u,)
      return
    raise RuntimeError('weird expression %s'%e)

  def compute(self,name,e,predefined=False):
    self.ensure(e)
    counts = defaultdict(lambda:0)
    def count(e):
      if e.is_numeric():
        return
      counts[e] += 1
      for arg in self.values[e][1]:
        count(arg)
    count(e)
    code = []
    env = {}
    def prec(desired,(prec,s)): # Precedences: +- = 0, * = 1, parentheses/symbol = 10
      if desired>prec:
        return '(%s)'%s
      return s
    def exp(e): # (precedence,string)
      if e.is_numeric():
        assert e.is_integer()
        if e.is_positive():
          return 10,str(e)
        return 1,str(e)
      if e in env:
        return 10,env[e]
      op,args = self.values[e]
      if not args:
        return 10,op
      if op.isalnum():
        return 10,'%s(%s)'%(op,','.join(exp(a)[1] for a in args))
      if op=='*' and self.mul and not args[0].is_numeric():
        return 10,'%s(%s)'%(self.mul,','.join(exp(a)[1] for a in args))
      if len(args)==1:
        assert op=='-'
        return 1,'-%s'%prec(1,exp(args[0]))
      a,b = map(cache,args)
      p = {'+':0,'-':0,'*':1}[op]
      a = prec(p,a)
      b = prec(p+(op=='-'),b)
      return p,'%s%c%s'%(a,op,b)
    def cache(e,name=None,predefined=False):
      r = exp(e)
      if (counts[e]==1 or r[1].isalnum() or e.is_numeric()) and not name:
        return r
      if not name:
        name = '%s%d'%(self.tmp,self.next)
        self.next += 1
      code.append('%s%s = %s;'%('' if predefined else 'const auto ',name,r[1]))
      env[e] = name
      return 10,name
    cache(e,name,predefined=predefined)
    return code

class Compiler(object):
  def __init__(self):
    self.header_lines = []
    self.source_lines = []
    self.forwards = []
    self.bodies = []

  def header(self,*lines):
    self.header_lines.extend(lines)

  def source(self,*lines):
    self.source_lines.extend(lines)

  def compile(self,predicate,d):
    name = predicate.__name__
    with scope('predicate %s'%name):
      body = []
      with scope('setup'):
        # Set up the symbolic space
        args = inspect.getargspec(predicate)[0]
        n = len(args)
        info('n = %d'%n)
        info('d = %d'%d)
        info('args = %s'%' '.join(args))
        C = [(SR.var(a+c),'%s.%c'%(a,c)) for a in args for c in 'xyz'[:d]]
        CR = PolynomialRing(ZZ,[a[0] for a in C],sparse=True)
        V = SR**d

        # Function header
        for line in predicate.__doc__.split('\n'):
          if line:
            line = '// '+line.strip()
            self.header(line)
            self.header(line)
        parameters = ', '.join('const int %si, const Vector<float,%d> %s'%(a,d,a) for a in args)
        signature = 'bool %s(%s)'%(name,parameters)
        degenerate_signature = 'static bool %s_degenerate(%s)'%(name,parameters)
        self.header('%s OTHER_EXPORT;\n'%signature)
        self.forwards.append(degenerate_signature+' OTHER_COLD OTHER_NEVER_INLINE;')
        body.append('%s {'%signature)

      # Evaluate constant term with interval arithmetic
      with scope('interval'):
        X = [V([C[i*d+j][0] for j in xrange(d)]) for i in xrange(n)]
        constant = predicate(*X)
        body.append('  // Evaluate with interval arithmetic first')
        body.append('  Interval filter;')
        body.append('  {')
        block = Block(CR,dict((v,'Interval(%s)'%s) for v,s in C),mul=None)
        body.append('\n'.join('    '+s for s in block.compute('filter',constant,predefined=True)))
        body.append('    if (OTHER_EXPECT(!filter.contains_zero(),true))\n      return filter.certainly_positive();')
        body.append('  }\n')

      # Evaluate constant term with integer arithmetic
      with scope('constant'):
        body.append('  // Fall back to integer arithmetic.  First we reevaluate the constant term.')
        body.append('  const Interval::Int %s;'%', '.join('%s(%s)'%(v,s) for v,s in C))
        body.append('  assert(%s);'%' && '.join('%s==%s'%(v,s) for v,s in C))
        block = Block(CR,dict((v,str(v)) for v,_ in C),mul='mul')
        body.append('\n'.join('  '+s for s in block.compute('pred',constant)))
        body.append('  assert(filter.contains(pred));')
        body.append('  if (OTHER_EXPECT(bool(pred),true))\n    return pred>0;\n')

      body.append('  // The constant term is exactly zero, so fall back to simulation of simplicity.')
      body.append('  return %s_degenerate(%s);'%(name,','.join('%si,%s'%(s,s) for s in args)))
      body.append('}\n')

      # Start degenerate function
      body.append(degenerate_signature+' {')

      body.append('  // Compute input permutation')
      body.append('  int order[%d] = {%s};'%(n,','.join('%si'%a for a in args)))
      body.append('  const int permutation = permutation_id(%d,order);\n'%n)

      body.append('  // Losslessly cast to integers')
      body.append('  OTHER_UNUSED const Interval::Int %s;\n'%', '.join('%s(%s)'%(v,s) for v,s in C))

      # Constant term is zero; add a different shift variable to each coordinate and expand as a polynomial.
      with scope('expand'):
        E = ['e%s'%v[0] for v in C]
        R = PolynomialRing(SR,E,sparse=True)
        E = R.gens()
        V = R**d
        Xe = [V([C[i*d+j][0]+E[i*d+j] for j in xrange(d)]) for i in xrange(n)]
        expansion = predicate(*Xe)

      # For now, the only simplification we do is to replace integers with +-1
      def simplify(e):
        if e.is_numeric():
          return sign(e)
        return e
      coefficients = map(simplify,expansion.coefficients())

      # Assign a unique-up-to-sign id to each coefficient in the expansion
      coef_to_id = Values(CR) # Map from coefficient to (id,sign)
      coef_to_id[SR(1)] = (0,1)
      id_to_coef = [SR(1)]
      for coef in coefficients:
        if coef in coef_to_id:
          pass
        elif -coef in coef_to_id:
          i,s = coef_to_id[-coef]
          coef_to_id[coef] = i,-s
        else:
          assert not coef.is_numeric()
          coef_to_id[coef] = len(id_to_coef),1 
          id_to_coef.append(coef)
      if 0:
        for i,c in enumerate(id_to_coef):
          info('id %d, coef %s'%(i,c))

      info('coefficients = %d, unique = %d'%(len(coefficients),len(id_to_coef)))
      body.append(('  // The constant term is zero, so we add infinitesimal shifts to each coordinate in the input, expand\n'
                  +'  // the result as a multivariate polynomial, and evaluate one term at a time until we hit a nonzero.\n'
                  +'  // Each coordinate gets a unique infinitesimal, each infinitely smaller than the last, so cancellation\n'
                  +'  // of all of them together is impossible.  In total, the error polynomial has %d terms, of which %d are\n'
                  +'  // unique (up to sign), but it usually suffices to evaluate only a few.\n')
                  %(len(coefficients),len(id_to_coef)))

      if 0:
        # From here on out, the result will depend on the ordering of the input arguments.  However, some of
        # these permutations produce the same answer with a possible sign flip.  Therefore, we organize the
        # set of all permutations into equivalence classes.
        with scope('permutations'):
          # Determine which permutations are identical or sign flipped versions of other permutations
          permutations = all_permutations(range(n))
          info('permutations = %d'%len(permutations))
          with scope('partition'):
            versions = Values(CR) # Map from expression to the representative permutation that generated it
            representatives = {} # If representatives[p] = q,s, f(X[p]) = s*f(X[q])
            for p in permutations:
              info('classifying p = %s'%(p,))
              pred = predicate(*subset(X,p))
              try:
                representatives[p] = versions[pred],1
              except KeyError:
                try:
                  representatives[p] = versions[-pred],-1
                except KeyError:
                  w = versions[pred] = len(versions),p
                  representatives[p] = w,1
            kinds = len(set(representatives.values()))
            info('distinct permutations = %d, kinds = %d'%(len(versions),kinds))
          # Generate table lookup code to map a permutation to its representative
          with scope('generate'):
            body.append('  // Determine which class of permutations we\'re in.  The lookup table is 2*representative+negate.')
            body.append('  // There %s %d different representative permutation%s, or %d counting differences in sign.'%('are' if len(versions)>1 else 'is',len(versions),'s' if len(versions)>1 else '',kinds))
            table = numpy.repeat(-1,factorial(n))
            for p in permutations:
              (r,_),s = representatives[p]
              table[selection_sort(list(p))] = 2*r+(s<0)
            body.append('  '+integer_table('canonicalize',table))
            body.append('  const int canonical = canonicalize[permutation];')
            if len(versions)>1:
              body.append('  const int representative = canonical>>1;')
            body.append('  const bool flip = canonical&1;\n')

      # In the degenerate case, the result depends on the ordering of the input arguments.  Thus, we loop over each
      # possible permutation and compute the necessary sequence of terms to evaluate.
      with scope('analyze'):
        permutations = sorted(all_permutations(range(n)),key=permutation_id)
        sequences = [None]*len(permutations)
        unordered = zip([coef_to_id[c] for c in coefficients],[numpy.asarray(m.degrees()).reshape(n,d) for m in expansion.monomials()])
        assert numpy.all(unordered[-1][1]==0)
        unordered = unordered[:-1]
        weights = numpy.int64(n)**numpy.arange(n*d).reshape(n,d)
        for ip,p in enumerate(permutations):
          info('%d/%d : %s'%(ip,len(permutations),p))
          inv_p = numpy.empty(len(p),dtype=int)
          inv_p[numpy.asarray(p)] = numpy.arange(len(p))
          # Sort monomials in lexicographic order using the reversed unpermuted order of the variables (since later variables are smaller)
          ordered = sorted(unordered,key=lambda m:numpy.tensordot(weights,m[1][inv_p]))
          # We'll need to compute until we hit a trivial nonzero
          sequence = []
          seen = set()
          for (i,s),_ in ordered:
            if i in seen:
              continue # We already know this coefficient is zero, so skip
            sequence.append((i,s))
            if i==0:
              break
            seen.add(i)
          else:
            raise NotImplemented('No monomials with constant coefficients: need to test the entire coefficient system for solvability')
          sequences[ip] = sequence

      with scope('generate'):
        if 0 and len(sequences)==1:
          body.append('  // All permutations produce the same predicate up to sign, so evaluation is straightforward.')
          for i,exp in enumerate(sequences[0]):
            if exp.is_numeric():
              body.append('  return %sflip;'%('!' if exp.is_positive() else ''))
            else:
              term = 'term%d'%i
              body.append('\n'.join('  '+s for s in block.compute(term,exp)))
              body.append('  if (%s) return flip^(%s>0);'%(term,term))
        else:
          body.append('  // Different permutations produce different predicates.  To reduce code size,\n' \
                     +'  // we use lookup tables and a switch statement.  I.e., a tiny bytecode interpreter.')
          terms = {0:0} # Map from unique coefficient id to its position in the switch statement
          tables = []
          with scope('terms'):
            for sequence in sequences:
              table = []
              for i,s in sequence:
                if i not in terms:
                  terms[i] = len(terms)
                table.append(2*terms[i]+(s<0))
              tables.append(table)
          body.append('  '+integer_table('starts',numpy.hstack([0,numpy.cumsum(map(len,tables))[:-1]])))
          body.append('  '+integer_table('terms',[t for table in tables for t in table]))
          body.append('  for (int i=starts[permutation];;i++) {')
          body.append('    const bool f = terms[i]&1;')
          body.append('    switch (terms[i]>>1) {')
          info('cases = %d'%len(terms))
          with scope('switch'):
            for c,i in sorted((c,i) for i,c in terms.iteritems()):
              e = id_to_coef[i]
              info('case %d = %s'%(c,e))
              if e.is_numeric():
                assert e==1
                body.append('      case %d:'%c)
                body.append('        return !f;')
              else:
                body.append('      case %d: {'%c)
                # We can reuse variables from the constant term, but not between any two cases in the switch statement.'
                body.append('\n'.join('        '+s for s in copy.deepcopy(block).compute('term',e)))
                body.append('        if (term) return f^(term>0); break; }')
          body.append('      default:\n        OTHER_UNREACHABLE();\n    }\n  }')
      body.append('}\n')
      self.bodies.append(body)

### Specific predicates

def triangle_oriented(p0,p1,p2):
  '''Is a 2D triangle positively oriented?'''
  return cross(p1-p0,p2-p0)

def segment_directions_oriented(a0,a1,b0,b1):
  '''Is the rotation from a1-a0 to b1-b0 positive?'''
  return cross(a1-a0,b1-b0)

def segment_intersections_ordered_helper(a0,a1,b0,b1,c0,c1):
  '''Given segments a,b,c, does intersect(a,b) come before intersect(a,c) on segment a?
  This predicate answers that question assuming that da,db and da,dc are positively oriented.'''
  da = a1-a0
  db = b1-b0
  dc = c1-c0
  return det(c0-a0,dc)*det(da,dc)-det(b0-a0,db)*det(da,db)

### Top level

if __name__=='__main__':
  compiler = Compiler()
  warning = '// Exact geometric predicates\n// Autogenerated by core/exact/sage/simplicity: DO NOT EDIT\n'
  compiler.header(warning+'#pragma once\n\n#include <other/core/vector/Vector.h>\nnamespace other {\n')
  compiler.source(warning+'\n#include <other/core/exact/predicates.h>\n#include <other/core/exact/Interval.h>\n#include <other/core/exact/Exact.h>\n'
    +'#include <other/core/exact/permutation_id.h>\n#include <algorithm>\nnamespace other {\n\nusing exact::mul;\nusing std::lower_bound;\n')

  # Compile predicates
  compiler.compile(triangle_oriented,2)
  compiler.compile(segment_directions_oriented,2)
  compiler.compile(segment_intersections_ordered_helper,2)

  # Finalize
  compiler.source('// Forward declarate degeneracy handling routines')
  compiler.source(*compiler.forwards)
  compiler.source('')
  for body in compiler.bodies:
    compiler.source(*body)
  compiler.header('}')
  compiler.source('}')

  # Write files
  os.chdir(os.path.dirname(sys.argv[0]))
  open('../predicates.h','w').write('\n'.join(compiler.header_lines)+'\n')
  open('../predicates.cpp','w').write('\n'.join(compiler.source_lines)+'\n')
