#!/usr/bin/env python
# Simulation of simplicity analysis and code generation

'''
Simulation of simplicity adds infinitesimal shifts to each coordinate input to an exact geometric
predicate.  Even if the original predicate (the constant term) is exactly zero, at least one of
the smaller terms of the polynomial will be nonzero, avoiding all degeneracies.  For details on
the original method, see

  Edelsbrunner and Mucke, http://arxiv.org/abs/math/9410209.

Unfortunately, the complexity of the expanded shift polynomial is exponential in the complexity
of the predicate.  Most of the time evaluating only a few terms will suffice, but code generation
is mandatory if we want to scale to more than a handful of simple predicates.  We use SymPy for
most of the necessary symbolic computation, though we also need to roll our own symbolic polynomial
class to prevent the coefficients from being unnecessarily expanded.

To regenerate the predicate code, run

    cd core/exact
    ./simplicity.py

The resulting predicates.{h,cpp} are checked into git so that users do not need SymPy to compile.
'''

from __future__ import with_statement
from collections import defaultdict
from contextlib import contextmanager
from polynomial import *
import inspect
import sympy
import numpy
import time
import sys
import os

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

def integer_table(name,table):
  TI = smallest_unsigned_integer(max(table)+1)
  return 'static const %s %s[%d] = {%s};'%(TI,name,len(table),','.join(map(str,table)))

### Basic linear algebra

dot = numpy.dot

def cross(u,v):
  if len(u)==len(v)==2:
    return u[0]*v[1]-u[1]*v[0]
  return numpy.cross(u,v)

def triple(u,v,w):
  a,b,c = m
  return u[0]*cross(v[1:],w[1:])+v[0]*cross(w[1:],u[1:])+w[0]*cross(u[1:],v[1:])

def det(*m):
  return (cross if len(m)==2 else triple)(*m)

### Code generation

class Values(object):
  def __init__(self,standardize):
    self.d = {}
    self.standardize = standardize

  def __len__(self):
    return len(self.d)

  def __getitem__(self,k):
    return self.d[self.standardize(k)]

  def __setitem__(self,k,v):
    self.d[self.standardize(k)] = v

  def __contains__(self,k):
    return self.standardize(k) in self.d

  def values(self):
    return self.d.values()

  def copy(self):
    v = Values(self.standardize)
    v.d = self.d.copy()
    return v

def remove_sign(e):
  if type(e)==sympy.Mul:
    if -1 in e.args:
      a = list(e.args)
      a.remove(-1)
      return -1,sympy.Mul(*a)
  return 1,e

class Block(object):
  def __init__(self,standardize,vars,mul,tmp='v'):
    self.vars = vars
    self.values = Values(standardize) # Map from expression to (op,args)
    for v,s in vars.iteritems():
      self.values[v] = s,()
    self.tmp = tmp
    self.next = 0
    self.mul = mul

  def copy(self):
    b = Block.__new__(Block)
    b.vars = self.vars
    b.values = self.values.copy()
    b.tmp = self.tmp
    b.next = self.next
    b.mul = self.mul
    return b

  def ensure(self,e):
    '''Ensure that either e is available'''
    values = self.values
    if e in values or e.is_Number:
      return
    if type(e)==sympy.Add:
      pos,neg = [],[]
      for t in e.args:
        s,t = remove_sign(t)
        (pos if s>0 else neg).append(t)
      if not pos:
        pos.append(-neg.pop())
      partial = pos[0]
      self.ensure(partial)
      for t in pos[1:]:
        self.ensure(t)
        u = partial
        partial = u+t
        values[partial] = '+',(u,t)
      for t in neg:
        self.ensure(t)
        u = partial
        partial = u-t
        values[partial] = '-',(u,t)
    elif type(e)==sympy.Mul:
      args = list(e.args) 
      if len(args)==2 and -1 in args:
        args.remove(-1)
        self.ensure(args[0])
        values[e] = '-',(args[0],)
      else:
        if -1 in args:
          args.remove(-1)
          args = [-args[0]]+args[1:]
        partial = args[0]
        self.ensure(partial)
        for t in args[1:]:
          self.ensure(t)
          u = partial
          partial = u*t
          values[partial] = '*',(u,t) 
    elif type(e)==sympy.Pow:
      u,v = e.args
      assert v.is_integer
      op = {2:'sqr',3:'cube'}[int(v)]
      self.ensure(u)
      values[e] = op,(u,)
    else:
      raise RuntimeError('weird expression %s'%e)

  def compute(self,name,e,predefined=False):
    self.ensure(e)
    counts = defaultdict(lambda:0)
    def count(e):
      if sympy.S(e).is_Number:
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
      if e.is_Number:
        assert e.is_integer
        if e.is_positive:
          return 10,str(e)
        return 1,str(e)
      if e in env:
        return 10,env[e]
      op,args = self.values[e]
      if not args:
        return 10,op
      if op.isalnum():
        return 10,'%s(%s)'%(op,','.join(exp(a)[1] for a in args))
      if op=='*' and self.mul and not args[0].is_Number:
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
      if (counts[e]==1 or r[1].isalnum() or e.is_Number) and not name:
        return r
      if not name:
        name = '%s%d'%(self.tmp,self.next)
        self.next += 1
      code.append('%s%s = %s;'%('' if predefined else 'const auto ',name,r[1]))
      env[e] = name
      return 10,name
    cache(e,name,predefined=predefined)
    return code

class CPPFunction:
  def __init__(self,name,parameter_string,body,return_type):
    self.name = name
    self.parameter_string = parameter_string
    self.body = body
    self.return_type = return_type
  
  def signature_string(self):
    # Build cpp function signature for this object
    return '%s %s(%s)'% (self.return_type, self.name, self.parameter_string)
  
  def implementation_lines(self):
    # Build cpp function body for this object
    result = []
    result.append(self.signature_string() + ' {')
    result += ['  ' + line for line in self.body]
    result.append('}\n')
    return result

class SimulatedSimplicityContext:
  def __init__(self,args,d):
    self.args = args
    self.arg_dims = d
    self.symbol_table = [(sympy.symbols(a+c),'%s.%c'%(a,c)) for a in args for c in 'xyz'[:d]]
    
  def standardize(self,e):
    # Map expression into sympy poly
    return sympy.Poly(e,tuple(v for v,s in self.symbol_table),domain=sympy.ZZ)

  def expand_with_symbols(self, fn):
    # evaluate expression with this objects symbols
    V = numpy.array
    X = [V([self.symbol_table[i*self.arg_dims+j][0] for j in xrange(self.arg_dims)]) for i in xrange(len(self.args))]
    return fn(*X)

  def expand_epsilons(self, fn):
    # When constant term is zero; add a different shift variable to each coordinate and expand as a polynomial.
    with scope('expand'):
      # degrees = set(map(sum,standardize(constant).monoms()))
      # info('degrees = %s'%list(degrees))
      # assert len(degrees)==1
      # degree, = degrees

      E = tuple(sympy.symbols('e%s'%v[0]) for v in self.symbol_table)
      ER = SymbolicPolynomialRing(E,self.standardize)
      E = ER.singletons()
      V = numpy.array
      Xe = [V([self.symbol_table[i*self.arg_dims+j][0]+E[i*self.arg_dims+j] for j in xrange(self.arg_dims)]) for i in xrange(len(self.args))]
      expansion = fn(*Xe).filter()
      # assert expansion.homogeneous()==degree
      return expansion

  def print_info(self):
    # Debugging function
    info('n = %d'%len(self.args))
    info('d = %d'%self.arg_dims)
    info('args = %s'%' '.join(self.args))

  def make_cpp_argument_list(self):
    # Generate argument list for cpp source files
    return ', '.join('const int %si, const Vector<float,%d> %s'%(a,self.arg_dims,a) for a in self.args)

  def implement_interval_function(self, name, e):
    # Create C++ function for evaluating expressions using interval arithmetic
    symbolic_expansion = self.expand_with_symbols(e)
    block = Block(self.standardize,dict((v,'Interval(%s)'%s) for v,s in self.symbol_table),mul=None)
    body = ['Interval result;'] + block.compute(name,symbolic_expansion,predefined=True) + ['return result;']
    return CPPFunction(name,self.make_cpp_argument_list(),body,'Interval')

  def implement_integer_function(self, name, e):
    # Create C++ function for evaluating expressions using integer arithmetic
    symbolic_expansion = self.expand_with_symbols(e)
    block = Block(self.standardize,dict((v,'Interval::Int(%s)'%str(v)) for v,_ in self.symbol_table),mul='mul')
    body = ['Interval::Int result;'] + block.compute(name,symbolic_expansion,predefined=True) + ['return result;']
    return CPPFunction(name,self.make_cpp_argument_list,body,'Interval::Int')

  def implement_perturbed_function(self, name, e):
    with scope('implement_polynomial_terms for %s' % name):
      expansion = self.expand_epsilons(e)
      unique_coefficients = set(expansion.terms.values())
      info('%d unique_coefficients:' % len(unique_coefficients))
      coef_to_id = {}
      for i,c in enumerate(unique_coefficients):
        coef_to_id[c] = i 
        info('[%d]: %s' % (i, str(c)))

      info('expansion: %s' % str(expansion))
      info('polynomial with %d terms:' % len(expansion.terms))
      for (exponents,coefficient) in expansion.terms.iteritems():
        info("[%s]: %d" % (str(exponents), coef_to_id[coefficient]))

      n = len(self.args)
      d = self.arg_dims
      
      permutations = sorted(all_permutations(range(n)),key=permutation_id)
      sequences = [None]*len(permutations)
      unordered = [(coef_to_id[c],numpy.asarray(deg).reshape(n,d)) for deg,c in expansion.terms.iteritems()]
      weights = numpy.int64(n)**numpy.arange(n*d).reshape(n,d)
      for ip,p in enumerate(permutations):
        info('%d/%d : %s'%(ip,len(permutations),p))
        inv_p = numpy.empty(len(p),dtype=int)
        inv_p[numpy.asarray(p)] = numpy.arange(len(p))
        # Sort monomials in lexicographic order using the reversed unpermuted order of the variables (since later variables are smaller)
        ordered = sorted(unordered,key=lambda m:numpy.tensordot(weights,m[1][inv_p]))
        # We'll need to compute until we hit a trivial nonzero
        sequence = [] # [coef_id for coef_id,exponents in ordered]
        seen = set()
        for (coef_id,exponents) in ordered:
          if coef_id in seen:
            continue # We already know this coefficient is zero, so skip
          sequence.append((coef_id,exponents))
          if exponents == zeros:
            break
          seen.add(i)
        else:
          raise NotImplemented('No monomials with constant coefficients: need to test the entire coefficient system for solvability')
        info(str(sequence))
        sequences[ip] = sequence


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

  def add_function(self, cpp_function):
    self.forwards += [cpp_function.signature_string() + ';']
    self.bodies.append(cpp_function.implementation_lines())

  def interval_body(self,C,n,d,standardize,constant):
    '''
    C = sympy symbol table
    n = number of arguments
    d = dimension of each argument
    standardize = converter from expression to sympy.Poly
    '''
    body = []
    # Evaluate constant term with interval arithmetic
    with scope('interval'):
      body.append('  // Evaluate with interval arithmetic first')
      body.append('  Interval filter;')
      body.append('  {')
      block = Block(standardize,dict((v,'Interval(%s)'%s) for v,s in C),mul=None)
      body.append('\n'.join('    '+s for s in block.compute('filter',constant,predefined=True)))
      body.append('    if (OTHER_EXPECT(!filter.contains_zero(),true))\n      return filter.certainly_positive();')
      body.append('  }\n')
    return body

  def constant_body(self,C,n,d,standardize,constant):
    # Evaluate constant term with integer arithmetic
    body = []
    with scope('constant'):
      body.append('  // Fall back to integer arithmetic.  First we reevaluate the constant term.')
      body.append('  OTHER_UNUSED const Interval::Int %s;'%', '.join('%s(%s)'%(v,s) for v,s in C))
      body.append('  assert(%s);'%' && '.join('%s==%s'%(v,s) for v,s in C))
      block = Block(standardize,dict((v,str(v)) for v,_ in C),mul='mul')
      body.append('\n'.join('  '+s for s in block.compute('pred',constant)))
      body.append('  assert(filter.contains(pred));')
      body.append('  if (OTHER_EXPECT(bool(pred),true))\n    return pred>0;\n')
    return body

  def degenerate_body(self,C,n,d,standardize,constant,predicate,args):
    body = []
    block = Block(standardize,dict((v,str(v)) for v,_ in C),mul='mul')
    body.append('  // Compute input permutation')
    body.append('  int order[%d] = {%s};'%(n,','.join('%si'%a for a in args)))
    body.append('  const int permutation = permutation_id(%d,order);\n'%n)

    body.append('  // Losslessly cast to integers')
    body.append('  OTHER_UNUSED const Interval::Int %s;\n'%', '.join('%s(%s)'%(v,s) for v,s in C))

    # For now, the only simplification we do is to replace integers with +-1
    def simplify(e):
      if sympy.S(e).is_Number:
        return numpy.sign(e)
      return e
    coefficients = map(simplify,expansion.terms.values())

    # Assign a unique-up-to-sign id to each coefficient in the expansion
    coef_to_id = Values(standardize) # Map from coefficient to (id,sign)
    coef_to_id[1] = (0,1)
    id_to_coef = [1]
    for coef in coefficients:
      if coef in coef_to_id:
        pass
      elif -coef in coef_to_id:
        i,s = coef_to_id[-coef]
        coef_to_id[coef] = i,-s
      else:
        assert not sympy.S(coef).is_Number
        coef_to_id[coef] = len(id_to_coef),1 
        id_to_coef.append(coef)

    info('coefficients = %d, unique = %d'%(len(coefficients),len(id_to_coef)))
    body.append(('  // The constant term is zero, so we add infinitesimal shifts to each coordinate in the input, expand\n'
                +'  // the result as a multivariate polynomial, and evaluate one term at a time until we hit a nonzero.\n'
                +'  // Each coordinate gets a unique infinitesimal, each infinitely smaller than the last, so cancellation\n'
                +'  // of all of them together is impossible.  In total, the error polynomial has %d terms, of which %d are\n'
                +'  // unique (up to sign), but it usually suffices to evaluate only a few.\n')
                %(len(coefficients),len(id_to_coef)))
    # In the degenerate case, the result depends on the ordering of the input arguments.  Thus, we loop over each
    # possible permutation and compute the necessary sequence of terms to evaluate.
    with scope('analyze'):
      permutations = sorted(all_permutations(range(n)),key=permutation_id)
      sequences = [None]*len(permutations)
      unordered = [(coef_to_id[simplify(c)],numpy.asarray(deg).reshape(n,d)) for deg,c in expansion.terms.iteritems() if any(deg)]
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
          if exp.is_Number:
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
            if sympy.S(e).is_Number:
              assert e==1
              body.append('      case %d:'%c)
              body.append('        return !f;')
            else:
              body.append('      case %d: {'%c)
              # We can reuse variables from the constant term, but not between any two cases in the switch statement.'
              body.append('\n'.join('        '+s for s in block.copy().compute('term',e)))
              body.append('        if (term) return f^(term>0); break; }')
        body.append('      default:\n        OTHER_UNREACHABLE();\n    }\n  }')
    return body

  def compile_predicate(self,predicate,d):
    name = predicate.__name__
    with scope('predicate %s'%name):
      body = []
      with scope('setup'):
        # Set up the symbolic space
        args = inspect.getargspec(predicate)[0]
        function = SimulatedSimplicityRational(args,d,predicate)
        
        # Function header
        for line in predicate.__doc__.split('\n'):
          if line:
            line = '// '+line.strip()
            self.header(line)

        degenerate_signature = 'static bool %s_degenerate(%s)'%(name,parameters)
        self.header('%s OTHER_EXPORT;\n'%signature)
        self.forwards.append(degenerate_signature+' OTHER_COLD OTHER_NEVER_INLINE;')
        body.append('%s {'%signature)
        V = numpy.array
        X = [V([C[i*d+j][0] for j in xrange(d)]) for i in xrange(n)]
        constant = predicate(*X)
      
      body += self.interval_body(C,n,d,standardize,constant)
      body += self.constant_body(C,n,d,standardize,constant)
      body.append('  // The constant term is exactly zero, so fall back to simulation of simplicity.')
      body.append('  return %s_degenerate(%s);'%(name,','.join('%si,%s'%(s,s) for s in args)))
      body.append('}\n')

      # Start degenerate function
      body.append(degenerate_signature+' {')
      body += self.degenerate_body(C,n,d,standardize,constant,predicate,args)
      body.append('}\n')
      self.bodies.append(body)

### Specific predicates

def rightwards(a,b):
  '''Is b.x > a.x?'''
  return b[0]-a[0]

def upwards(a,b):
  '''Is b.y > a.y?'''
  return b[1]-a[1]

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
  warning = '// Exact geometric predicates\n// Autogenerated by core/exact/simplicity: DO NOT EDIT\n'
  compiler.header(warning+'#pragma once\n\n#include <other/core/vector/Vector.h>\nnamespace other {\n')
  compiler.source(warning+'\n#include <other/core/exact/predicates.h>\n#include <other/core/exact/Interval.h>\n#include <other/core/exact/Exact.h>\n'
    +'#include <other/core/exact/permutation_id.h>\n#include <algorithm>\nnamespace other {\n\nusing exact::mul;\nusing std::lower_bound;\n')

  test_fn = triangle_oriented
  test_name = test_fn.__name__
  ssc = SimulatedSimplicityContext(inspect.getargspec(test_fn)[0], 2)
  
  compiler.add_function(ssc.implement_interval_function("interval_%s" % test_name, test_fn))
  compiler.add_function(ssc.implement_integer_function("integer_%s" % test_name, test_fn))

  ssc.implement_perturbed_function("degenerate_%s" % test_name, test_fn)
  # Compile predicates
  # compiler.compile(rightwards,2)
  # compiler.compile(upwards,2)
  # compiler.compile(triangle_oriented,2)
  # compiler.compile(segment_directions_oriented,2)
  # compiler.compile(segment_intersections_ordered_helper,2)

  # Finalize
  compiler.source('// Forward declare degeneracy handling routines')
  compiler.source(*compiler.forwards)
  compiler.source('')
  for body in compiler.bodies:
    compiler.source(*body)
  compiler.header('}')
  compiler.source('}')

  # Write files
  os.chdir(os.path.dirname(sys.argv[0]))
  open('new_predicates.h','w').write('\n'.join(compiler.header_lines)+'\n')
  open('new_predicates.cpp','w').write('\n'.join(compiler.source_lines)+'\n')
