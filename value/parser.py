import re
import sys
import argparse
from numpy import *
from other.core import *
from other.core.utility import Log
from other.core.vector import *

def fix_name(name):
  return name.replace('_','-')

def try_autocomplete():
  import os
  def return_options(options, include_files=False):
    os.system("echo Completion_options_start")
    os.system("compgen %s -W \"%s\" -- \"%s\"" % ("-f" if include_files else "", options, curr))
    os.system("echo Completion_options_end")
    exit(0)
  if len(sys.argv) != 4 or sys.argv[1] != "--autocomplete": return

  print "Autocomplete requested for %s" % sys.argv
  prev = sys.argv[2]
  curr = sys.argv[3]

  if len(prev) > 2 and prev[:2] == "--": prev = prev[2:]
  prev = prev.replace('-','_')

  print "prev: %s" % prev
  if prev in PropManager.items():
    prop = PropManager.items()[prev]
    t = type(prop())
    if len(prop.allowed) > 0:
      return_options(" ".join(prop.allowed))
    elif t==bool:
      return_options("0 no false 1 yes true")
    elif t==str:
      return_options("", include_files=True)
    else:
      return_options("", include_files=False)
      
  return_options(" ".join(["--"+fix_name(n) for n in PropManager.items().keys()]), include_files=True)
          
trailing_pattern = re.compile(r'0000.*')

def format_number(v):
  if type(v) in (int,int32,int64):
    return str(v)
  try:
    if int(v)==v:
      return str(int(v))
    if v>0:
      e = int(round(log(v)/log(10)))
      s = '1e%d'%e
      if float(s)==v and abs(e)>2:
        return s
  except OverflowError:
    pass
  r = repr(v)
  reduced = trailing_pattern.sub('',r)
  try:
    vr = float(reduced)
    if vr==v:
      return reduced
  except ValueError:
    pass
  return r

def format_commas(v):
  return ','.join(map(format_number,v.ravel()))

def format_value(prop):
  v = prop()
  t = type(v)
  if t==bool:
    return '01'[v]
  elif t==str:
    return v
  elif t in (int,int32,int64,float,float32,float64):
    return format_number(v)
  elif t==ndarray:
    return format_commas(v)
  elif t==Rotation.Rotations3d or t==Frame.Frames:
    return format_commas(v.reals())
  else:
    raise NotImplementedError("Don't know how to convert value %r of type %s to a command line string"%(v,t))
          
def parse(description,positional=[]):
  parser = argparse.ArgumentParser(description = description)

  try_autocomplete()
  
  props_set = set()
  positional = [(p.name if isinstance(p,Value) else p) for p in positional]

  class PropAction(argparse.Action):
    def __call__(self,parser,namespace,values,option_string=None):
      if values is not None:
        name = self.dest.replace('-','_')
        prop = PropManager.get(name)
        try:
          prop.set(values)
        except ValueError:
          raise ValueError("property '%s' expects one of '%s', got '%s'"%(self.dest,' '.join(map(str,prop.allowed)),values))
        props_set.add(name)

  bools = dict((s,bool(i)) for i,ss in enumerate(('0 no false','1 yes true')) for s in ss.split())
  def convert_bool(s):
    return bools[s.lower()]

  def convert_list(s):
    return ''.join(s)

  def convert_rotation3d(s):
    sv = map(float,s.split(','))
    assert len(sv)==4
    sv = normalized(sv)
    return Rotation.from_sv(sv[0],sv[1:])

  def convert_frame3d(s):
    f = map(float,s.split(','))
    assert len(f)==7
    sv = normalized(f[3:])
    return Frames(f[:3],Rotation.from_sv(sv[0],sv[1:]))

  def convert_vector(s):
    return fromstring(s,sep=',')

  def converter(prop):
    v = prop()
    t = type(v)
    if t==bool:
      return convert_bool
    elif t==Rotation.Rotations3d:
      return convert_rotation3d
    elif t==Frame.Frames:
      assert v.d==3
      return convert_frame3d
    elif t==ndarray:
      assert len(v.shape)==1
      return convert_vector
    elif t==list:
      return convert_list
    else:
      return t

  def help(prop):
    help = prop.help
    if not help:
      return None
    if '%default' in help:
      help = help.replace('%default',format_value(prop))
    return help

  # Add keyword arguments
  for name,prop in PropManager.items().items():
    # good arguments to consider adding to properties:
    # help

    # this should work ok for simple types, for complex types (lists, etc.), we need special processing
    options = ['--'+fix_name(name)]
    if prop.abbrev:
      options.append('-'+prop.abbrev)
    parser.add_argument(*options,
                        required=(prop.required and name not in positional),
                        type=converter(prop),
                        help=help(prop),
                        action=PropAction)

  # Add positional arguments
  for name in positional:
    prop = PropManager.get(name)
#NEEDS A CHECK HERE, AS SOME POSITIONALS FAIL (e.g. solar/layout)
    parser.add_argument(fix_name(name),nargs='*' if type(prop())==list else '?',type=converter(prop),action=PropAction)

  # Avoid jumbled output if we're inside a Log scope
  Log.flush()

  # Parse the input
  parser.parse_args()

  # Check for missing positional arguments
  for name in positional:
    prop = PropManager.get(name)
    if prop.required and name not in props_set:
      parser.error('missing positional argument %s'%fix_name(name))

  # Verify that we're able to convert the current prop values back into a command line.
  # It's very annoying to go through a whole bunch of setup only to fine that a bug
  # presents one from saving work.
  command(drop_defaults=True)
  command(drop_defaults=False)

  return props_set

def command(drop_defaults=True):
  args = [sys.argv[0]]
  for name in PropManager.order():
    prop = PropManager.get(name)
    if not (drop_defaults and all(prop()==prop.default)) and not prop.hidden:
      try:
        v = format_value(prop)
        if v.startswith('-'):
          args.append('--%s=%s'%(fix_name(name),v))
        else:
          args.extend(['--'+fix_name(name),v])
      except NotImplementedError:
        print 'Warning: Could not convert property %s, command line may be incomplete.' % name.replace('_','-')

  def escape(s):
    if ' ' in s or s=='':
      return "'%s'"%s
    return s

  return ' '.join(map(escape,args))
