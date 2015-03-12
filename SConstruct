import os
import re
import sys
import glob
import subprocess

try:
    from shlex import quote # For Python 3.3+
except ImportError:
    from pipes import quote # Python 2.7

def die(message):
  print>>sys.stderr,'fatal:',message
  sys.exit(1)

# Require a recent version of scons
EnsureSConsVersion(2,0,0)

# Read configuration from config.py.
# As far as I can tell, the scons version of this doesn't allow defining new options in SConscript files.
sys.path.insert(0,Dir('#').abspath)

try:
  import config
  has_config = True
except ImportError:
  has_config = False

del sys.path[0]
def options(env,*vars):
  for name,help,default in vars:
    Help('%s: %s (default %s)\n'%(name,help,default))
    if name in ARGUMENTS:
      value = ARGUMENTS[name]
      try:
        value = int(value)
      except ValueError:
        if ',' in value:
          value = value.split(',')
        else:
          pass
      env[name] = value
    elif has_config:
      env[name] = config.__dict__.get(name,default)
    else:
      env[name] = default

# Base environment
env = Environment(tools=['default','pdflatex','pdftex','textfile'])
default_cxx = env['CXX']
posix = env['PLATFORM']=='posix'
darwin = env['PLATFORM']=='darwin'
windows = env['PLATFORM']=='win32'
env.Replace(ENV=os.environ)
verbose = True

# Default base directory
if darwin:
  for base in '/opt/local','/usr/local':
    if os.path.exists(base):
      env.Replace(base=base)
      env.Append(CPPPATH=[base+'/include'],LIBPATH=[base+'/lib'])
      break
  else:
    die("We're on Mac, but neither /opt/local or /usr/local exists.  Geode requires either MacPorts or Homebrew.")
else:
  env.Replace(base='/usr')

# Add install or develop targets only if explicitly asked for on the command line
env['install'] = 'install' in COMMAND_LINE_TARGETS
env['develop'] = 'develop' in COMMAND_LINE_TARGETS
if env['develop']:
  env.Alias('develop',[])

# Base options
options(env,
  ('cxx','C++ compiler','<detect>'),
  ('arch','Architecture (e.g. opteron, nocona, powerpc, native)','native'),
  ('type','Type of build (e.g. release, debug, profile)','release'),
  ('default_arch','Architecture that doesn\'t need a suffix',''),
  ('cache','Cache directory to use',''),
  ('shared','Build shared libraries',1),
  ('shared_objects','Build shareable objects when without shared libraries',1),
  ('real','Primary floating point type (float or double)','double'),
  ('install_programs','install programs into source directories',1),
  ('Werror','turn warnings into errors',1),
  ('Wconversion','warn about various conversion issues',0),
  ('hidden','make symbols invisible by default',0),
  ('openmp','use openmp',1),
  ('syntax','check syntax only',0),
  ('use_latex','Use latex (mostly for internal technical documents)',0),
  ('thread_safe','use thread safe reference counting in pure C++ code',1),
  ('optimizations','override default optimization settings','<default>'),
  ('sse','Use SSE if available',1),
  ('skip','list of modules to skip',[]),
  ('skip_libs', 'list of libraries to skip', []),
  ('skip_programs','Build libraries only',0),
  ('base','Standard base directory for headers and libraries',env['base']),
  ('cxxflags_extra','',[]),
  ('linkflags_extra','',[]),
  ('cpppath_extra','',['/usr/local/include']),
  ('libpath_extra','',['/usr/local/lib']),
  ('rpath_extra','',[]),
  ('libs_extra','',[]),
  ('prefix','Path to install libraries, binaries, and scripts','/usr/local'),
  ('prefix_include','Override path to install headers','$prefix/include'),
  ('prefix_lib','Override path to install libraries','$prefix/lib'),
  ('prefix_bin','Override path to install binaries','$prefix/bin'),
  ('prefix_share','Override path to install resources','$prefix/share'),
  ('boost_lib_suffix','Suffix to add to each boost library','-mt'),
  ('python','Python executable','python'),
  ('mpicc','MPI wrapper compiler (used only to extract flags)','<detect>'),
  ('qtdir','Top level Qt dir (autodetect by default)',''))
assert env['real'] in ('float','double')

# Make a pristine environment for latex use
latex_env = env.Clone()

# Extra flag options
env.Append(CXXFLAGS=env['cxxflags_extra'])
env.Append(LINKFLAGS=env['linkflags_extra'])
env.Append(CPPPATH=env['cpppath_extra'])
env.Append(LIBPATH=env['libpath_extra'])
env.Append(RPATH=env['rpath_extra'])
env.Append(LIBS=env['libs_extra'])

# Improve performance
env.Decider('MD5-timestamp')
env.SetOption('max_drift',100)
env.SetDefault(CPPPATH_HIDDEN=[]) # directories in CPPPATH_HIDDEN won't be searched for dependencies
env.SetDefault(CPPDEFINES=[])
env.Replace(_CPPINCFLAGS=env['_CPPINCFLAGS']+re.sub(r'\$\( (.*)\bCPPPATH\b(.*)\$\)',r'\1CPPPATH_HIDDEN\2',env['_CPPINCFLAGS']))

# Pick compiler if the user requested the default
if env['cxx']=='<detect>':
  if windows:
    env.Replace(cxx=default_cxx)
  else:
    for gcc in 'clang++ g++-4.7 g++-4.6 g++'.split():
      if subprocess.Popen(['which',gcc], stdout=subprocess.PIPE).communicate()[0]:
        env.Replace(cxx=gcc)
        break
    else:
      die('no suitable version of g++ found')
env.Replace(CXX=env['cxx'])

# If we're using gcc, insist on 4.6 or higher
if re.match(r'\bg\+\+',env['cxx']):
  version = subprocess.Popen([env['cxx'],'--version'], stdout=subprocess.PIPE).communicate()[0]
  m = re.search(r'\s+([\d\.]+)(\s+|\n|$)',version)
  if not m:
    die('weird version line: %s'%version[:-1])
  version_tuple = tuple(map(int, m.group(1).split('.')))
  if version_tuple<(4,6):
    die('gcc 4.6 or higher is required, but %s has version %s'%(env['cxx'],m.group(1)))
  if version_tuple[0:2] == (4,7) and version_tuple[2] in (0,1):
    die('use of gcc 4.7.0 or 4.7.1 is strongly discouraged (ABI incompatabilities when mixing C++11 and other standards). %s is version %s'%(env['cxx'],m.group(1)))

# Build cache
if env['cache']!='':
  CacheDir(env['cache'])

# Make a variant-independent environment for building python modules
python_env = env.Clone(SHLIBPREFIX='',LINK=env['cxx'])
if darwin:
  python_env.Replace(LDMODULESUFFIX='.so',SHLIBSUFFIX='.so')

# Variant build setup
env['variant_build'] = os.path.join('build',env['arch'],env['type'])
env.VariantDir(env['variant_build'],'.',duplicate=0)

# Compiler flags
clang = bool(re.search(r'clang\b',env['cxx']))
if windows:
  if env['arch'] != 'native':
    raise RuntimeError("Specific architecture configuration on windows not supported")
    # Warning: If adding more options for architecture, be aware that /arch:AVX seems to have some compiler bugs with the current VS2015 preview (as of Feburary 2015)
  # Since I don't know a sane way to detect which target is active I'm leaving this commented out for now:
  # if env['sse'] and building_for_32bit_target():
  #  # If this give an 'unrecognized option' warning you might be using the compiler for 64 bit targets which has implied support for SSE
  #  env.Append(CXXFLAGS=' /arch:SSE2')
  if env['type'] == 'debug':
    env.Append(CXXFLAGS=' /Od /Oy-')
  if env['type'] in ('debug','optdebug'):
    env.Append(CXXFLAGS=' /Zi /FS')
    # You might try this instead if your debugger is having trouble locating the pdb files:
    #   env.Append(CXXFLAGS=' /Z7',LINKFLAGS=' /DEBUG')
  if env['type'] in ('release','optdebug','profile'):
    env.Append(CXXFLAGS=' /O2')
  env.Append(CXXFLAGS=' /EHs /W3',LINKFLAGS='/ignore:4221')
  if env['cxx'].endswith('icl') or env['cxx'].endswith('icl"'):
    env.Append(CXXFLAGS=' /wd2415 /wd597 /wd177')
  if env['type']=='debug':
    env.Append(CXXFLAGS=' /RTC1 /MDd',CCFLAGS=' /MDd',LINKFLAGS=' /DEBUG')
  else:
    env.Append(CXXFLAGS=' /MD')
  #dangerous: env.Append(LINKFLAGS='/NODEFAULTLIB:libcmtd.lib')
elif env['cxx'].endswith('icc') or env['cxx'].endswith('icpc'):
  if env['type']=='optdebug' or env['type']=='debug':
    env.Append(CXXFLAGS=' -g')
  if env['type']=='release' or env['type']=='optdebug' or env['type']=='profile':
    env.Append(CXXFLAGS=' -O3')
  env.Append(CXXFLAGS=' -w -vec-report0 -Wall -Winit-self -Woverloaded-virtual',LINKFLAGS=' -w')
else: # assume g++...
  gcc = True
  # Machine flags
  def ifsse(s):
    # return s if env['sse'] else ' -mno-sse'
    return s if env['sse'] else ''
  if env['arch']=='athlon':    machine_flags = ' -march=athlon-xp '+ifsse('-msse')
  elif env['arch']=='nocona':  machine_flags = ' -march=nocona '+ifsse('-msse2')
  elif env['arch']=='opteron': machine_flags = ' -march=opteron '+ifsse('-msse3')
  elif env['arch']=='corei7':  machine_flags = ' -march=corei7 '
  elif env['arch']=='powerpc': machine_flags = ''
  elif env['arch']=='native':  machine_flags = ' -march=native -mtune=native '+ifsse('')
  else: machine_flags = ''
  env.Append(CXXFLAGS=machine_flags)
  # Type specific flags
  if env['type']=='optdebug': env.Append(CXXFLAGS=' -g3')
  if env['type']=='release' or env['type']=='optdebug' or env['type']=='profile':
    optimizations = env['optimizations']
    if optimizations=='<default>':
      if   env['arch']=='pentium4': optimizations = '-O2 -fexpensive-optimizations -falign-functions=4 -funroll-loops -fprefetch-loop-arrays'
      elif env['arch']=='pentium3': optimizations = '-O2 -fexpensive-optimizations -falign-functions=4 -funroll-loops -fprefetch-loop-arrays'
      elif env['arch']=='opteron':  optimizations = '-O2'
      elif env['arch'] in ('nocona','native','powerpc', 'corei7'): optimizations = '-O3 -funroll-loops'
    env.Append(CXXFLAGS=optimizations)
    if not clang:
      env.Append(LINKFLAGS=' -dead_strip')
    if env['type']=='profile': env.Append(CXXFLAGS=' -pg',LINKFLAGS=' -pg')
  elif env['type']=='debug': env.Append(CXXFLAGS=' -g',LINKFLAGS=' -g')
  env.Append(CXXFLAGS=' -Wall -Winit-self -Woverloaded-virtual -Wsign-compare -fno-strict-aliasing') # -Wstrict-aliasing=2

# Optionally warn about conversion issues
if env['Wconversion']:
  env.Append(CXXFLAGS='-Wconversion -Wno-sign-conversion')

# Optionally stop after syntax checking
if env['syntax']:
  assert clang or gcc
  env.Append(CXXFLAGS='-fsyntax-only')
  # Don't rebuild files if syntax checking succeeds the first time
  for rule in 'CXXCOM','SHCXXCOM':
    env[rule] += ' && touch $TARGET'

# Use c++11
if not windows:
  if darwin:
    env.Append(CXXFLAGS=' -std=c++11')
  else: # Ubuntu
    env.Append(CXXFLAGS=' -std=c++0x')

# Hide symbols by default if desired
if env['hidden']:
  env.Append(CXXFLAGS=' -fvisibility=hidden',LINKFLAGS=' -fvisibility=hidden')

if env['Werror']:
  env.Append(CXXFLAGS=(' /WX' if windows else ' -Werror'))

# Relax a few warnings for clang
if clang:
  env.Append(CXXFLAGS=' -Wno-array-bounds -Wno-unknown-pragmas -Wno-deprecated') # for Python and OpenMP, respectively

# Decide whether to disable assertions (does not affect GEODE_ASSERT)
if env['type']=='release' or env['type']=='profile' or env['type']=='optdebug':
  env.Append(CPPDEFINES=['NDEBUG'])

# Enable OpenMP
if env['openmp']:
  if windows:
    env.Append(CXXFLAGS=' /openmp')
  elif not clang:
    env.Append(CXXFLAGS='-fopenmp',LINKFLAGS='-fopenmp')
  else:
    print>>sys.stderr, 'Warning: clang doesn\'t know how to do OpenMP, so many things will be slower'

# Work around apparent bug in variable expansion
env.Replace(prefix_lib=env.subst(env['prefix_lib']))

# External libraries
externals = {}
lazy_externals = {}

def external(env,name,lazy=1,**kwargs):
  '''Add an external library.  See external_helper for keyword argument details.'''
  if name in externals or name in lazy_externals:
    raise RuntimeError("Trying to redefine the external %s"%name)

  def fail():
    env['use_'+name] = 0
    if name in externals:
      del externals[name]
    if name in lazy_externals:
      del lazy_externals[name]
    if kwargs.get('required'):
      print>>sys.stderr, 'FATAL: %s is required.  See config.log for error details.'%name
      Exit(1)

  # Do we want to use this external?
  Help('\n')
  options(env,('use_'+name,'Use '+name+' if available',1))
  if env['use_'+name]:
    # Remember for lazy configuration
    lazy_externals[name] = env,kwargs,fail
    env['need_'+name] = kwargs.get('default',False)
  else:
    fail()

  # Some externals can't be lazy
  if not lazy:
    force_external(name)

def has_external(name):
  if name in externals:
    return True
  force_external(name)
  return name in externals

def force_external(name):
  try:
    env,kwargs,fail = lazy_externals[name]
  except KeyError:
    return
  external_helper(env,name,fail=fail,**kwargs)

def external_helper(env,name,default=0,dir=0,flags=(),cxxflags='',linkflags='',cpppath=(),libpath=(),rpath=0,libs=(),publiclibs=(),
             copy=(),frameworkpath=(),frameworks=(),requires=(),hide=False,callback=None,
             headers=None,configure=None,preamble=(),body=(),required=False,fail=None):
  for r in requires:
    force_external(r)
    if not env['use_'+r]:
      if verbose:
        print 'disabling %s: no %s'%(name,r)
      fail()
      return

  lib = {'dir':dir,'flags':flags,'cxxflags':cxxflags,'linkflags':linkflags,'cpppath':cpppath,'libpath':libpath,
         'rpath':rpath,'libs':libs,'copy':copy,'frameworkpath':frameworkpath,'frameworks':frameworks,'requires':requires,
         'publiclibs':publiclibs,'hide':hide,'callback':callback,'name':name}
  del lazy_externals[name]
  externals[name] = lib

  # Make sure empty lists are copied and do not refer to the same object
  for n in lib:
    if lib[n] == ():
      lib[n] = []

  Help('\n')
  options(env,
    (name+'_dir','Base directory for '+name,dir),
    (name+'_include','Include directory for '+name,0),
    (name+'_libpath','Library directory for '+name,0),
    (name+'_rpath','Extra rpath directory for '+name,0),
    (name+'_libs','Libraries for '+name,0),
    (name+'_publiclibs','Public (re-exported) libraries for '+name,0),
    (name+'_copy','Copy these files to the binary output directory for ' + name,0),
    (name+'_frameworks','Frameworks for '+name,0),
    (name+'_frameworkpath','Framework path for '+name,0),
    (name+'_cxxflags','Compiler flags for '+name,0),
    (name+'_linkflags','Linker flags for '+name,0),
    (name+'_requires','Required libraries for '+name,0),
    (name+'_pkgconfig','pkg-config names for '+name,0),
    (name+'_callback','Arbitrary environment modification callback for '+name,0))

  # Absorb settings
  if env[name+'_pkgconfig']!=0: lib['pkg-config']=env[name+'_pkgconfig']
  if 'pkg-config' in lib and lib['pkg-config']:
    def pkgconfig(pkg,data):
      return subprocess.Popen(['pkg-config',pkg,data],stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].replace('\n','')
    pkg = lib['pkg-config']
    includes = pkgconfig(pkg,"--cflags-only-I").split()
    lib['cpppath'] = [x.replace("-I","") for x in includes]
    lib['cxxflags'] = pkgconfig(pkg,"--cflags-only-other")
    lib['linkflags'] = pkgconfig(pkg,"--libs")
  dir = env[name+'_dir']
  def sanitize(path):
    return [path] if isinstance(path,str) else path
  if env[name+'_include']!=0: lib['cpppath'] = sanitize(env[name+'_include'])
  elif dir and not lib['cpppath']: lib['cpppath'] = [dir+'/include']
  if env[name+'_libpath']!=0: lib['libpath'] = sanitize(env[name+'_libpath'])
  elif dir and not lib['libpath']: lib['libpath'] = [dir+'/lib']
  if env[name+'_rpath']!=0: lib['rpath'] = sanitize(env[name+'_rpath'])
  elif lib['rpath']==0: lib['rpath'] = [Dir(d).abspath for d in lib['libpath']]
  if env[name+'_libs']!=0: lib['libs'] = env[name+'_libs']
  if env[name+'_publiclibs']!=0: lib['publiclibs'] = env[name+'_publiclibs']
  if env[name+'_frameworks']!=0: lib['frameworks'] = env[name+'_frameworks']
  if env[name+'_frameworkpath']!=0: lib['frameworkpath'] = sanitize(env[name+'_frameworkpath'])
  if env[name+'_cxxflags']!=0: lib['cxxflags'] = env[name+'_cxxflags']
  if env[name+'_linkflags']!=0: lib['linkflags'] = env[name+'_linkflags']
  if env[name+'_copy']!=0: lib['copy'] = env[name+'_copy']
  if env[name+'_callback']!=0: lib['callback'] = env[name+'_callback']

  # Check whether the external is usable
  if configure is not None:
    has = configure if isinstance(configure,bool) else configure(env,lib)
    if not has:
      fail()
  else:
    assert headers is not None

    # Match the configure environment to how we'll actually build things
    env_conf = env.Clone()
    for n in externals.keys():
      env_conf['need_'+n] = 0
    env_conf['need_'+name] = 1
    env_conf = link_flags(env_conf)
    libraries = [externals[n] for n in externals.keys() if env_conf.get('need_'+n)]
    def absorb(source,**kwargs):
      for k,v in kwargs.iteritems():
        env_conf[k] = v
    objects_helper(env_conf,'',libraries,absorb)

    # Check whether the library is usable
    def check(context):
      context.Message('checking for %s: '%name)
      source = '\n'.join(list(preamble)+['#include <%s>'%h for h in headers]+['int main() {']+list(body)+['  return 0;','}\n'])
      r = context.TryLink(source,extension='.cpp')
      context.Result(r)
      return r
    conf = env_conf.Configure(custom_tests={'Check':check})
    if not conf.Check():
      fail()
    conf.Finish()

# Library configuration.  Local directories must go first or scons will try to install unexpectedly
env.Prepend(CPPPATH=['#.','#$variant_build'])
env.Append(CPPPATH=[env['prefix_include']],LIBPATH=[env['prefix_lib']])
env.Prepend(LIBPATH=['#$variant_build/lib'])
if posix:
  env.Append(LINKFLAGS='-Wl,-rpath-link=$variant_build/lib')

# Account for library dependencies
def add_dependencies(env):
  libs = []
  for name in tuple(lazy_externals.iterkeys()):
    if env.get('need_'+name):
      force_external(name)
  for name,lib in externals.iteritems():
    if env.get('need_'+name):
      libs += lib['requires']
  while libs:
    name = libs.pop()
    env['need_'+name] = 1
    libs.extend(externals[name]['requires'])

# Linker flags
all_uses = []
def link_flags(env):
  if not windows:
    env_link = env.Clone(LINK=env['cxx'])
  else:
    env_link = env.Clone()
  add_dependencies(env_link)
  workaround = env.get('need_qt',0)

  for name,lib in externals.items():
    if env_link.get('need_'+name):
      all_uses.append('need_'+name)
      env_link.Append(LINKFLAGS=lib['linkflags'],LIBS=lib['libs'],FRAMEWORKPATH=lib['frameworkpath'],FRAMEWORKS=lib['frameworks'])
      if darwin:
        env_link.Append(LINKFLAGS=' '.join('-Xlinker -reexport-l%s'%n for n in lib['publiclibs']))
      else:
        env_link.Append(LIBS=lib['publiclibs'])
      env_link.AppendUnique(LIBPATH=lib['libpath'])
      if workaround: # Prevent qt tool from dropping include paths when building moc files
        env_link.PrependUnique(CPPPATH=lib['cpppath'])
      if lib.has_key('rpath'):
        env_link.PrependUnique(RPATH=lib['rpath'])
      if lib['callback'] is not None:
        lib['callback'](env_link)
  return env_link

# Copy necessary files into bin (necessary for dlls on windows)
copied_files = set()
def copy_files(env):
  env_copy = env.Clone()
  add_dependencies(env_copy)
  for name,lib in externals.items():
    if env['need_'+name]:
      for cp in lib['copy']:
        target = env['prefix_bin']+cp
        if target not in copied_files:
          copied_files.add(target)
          env_copy.Install(env['prefix_bin'], cp)
  return env_copy

# Convert sources into objects
def objects_helper(env,source,libraries,builder):
  if type(source)!=str: return source # assume it's already an object
  cpppath_reversed = env['CPPPATH'][::-1]
  cpppath_hidden_reversed = env['CPPPATH_HIDDEN'][::-1]
  if darwin:
    frameworkpath = env['FRAMEWORKPATH'][:]
    frameworks = env['FRAMEWORKS'][:]
  else:
    frameworks,frameworkpath = [],[]
  cxxflags = str(env['CXXFLAGS'])
  for lib in libraries:
    (cpppath_hidden_reversed if lib['hide'] else cpppath_reversed).extend(lib['cpppath'][::-1])
    frameworkpath.extend(lib['frameworkpath'])
    frameworks.extend(lib['frameworks'])
    cxxflags += lib['cxxflags']
  return builder(source,CPPPATH=cpppath_reversed[::-1],CPPPATH_HIDDEN=cpppath_hidden_reversed[::-1],FRAMEWORKPATH=frameworkpath,FRAMEWORKS=frameworks,CXXFLAGS=cxxflags)
def objects(env,sources):
  add_dependencies(env)
  libraries = [externals[name] for name in externals.keys() if env['need_'+name]]
  for lib in libraries:
    if lib['callback'] is not None:
      env = env.Clone()
      lib['callback'](env)
  builder = env.SharedObject if env['shared_objects'] or env['shared'] else env.StaticObject
  if type(sources)==list: return [objects_helper(env,source,libraries,builder) for source in sources]
  else: return objects_helper(env,sources,libraries,builder)

# Recursively list all files beneath a directory
default_skip = frozenset(['build','Debug','DebugStatic (Otherlab)','Release','ReleaseDLL (Otherlab)'])
def files(dir,skip=()):
  for f in os.listdir(dir):
    if f.startswith('.') or f in default_skip or f in skip:
      continue
    df = os.path.join(dir,f)
    if os.path.isdir(df):
      for c in files(df,skip):
        fc = os.path.join(f,c)
        if fc not in skip:
          yield fc
    yield f

# Automatic generation of library targets
if windows:
  all_libs = []
  python_libs = []
  projects = []

def dir_or_file(path):
  return Entry(path).disambiguate() # either File or Dir

# Target must be a directory!
def install_or_link(env,target,src):
  if env['install'] or env['develop']:
    # Get the real location of src
    srcpath = dir_or_file(src).abspath
    if not os.path.exists(srcpath):
      srcpath = dir_or_file(src).srcnode().abspath
      if not os.path.exists(srcpath):
        raise RuntimeError("can't find %s in either source or build directories"%src)
    if env['install']:
      env.Alias('install',env.Install(target,srcpath))
    elif env['develop']:
      env.Execute("d='%s' && mkdir -p $$$$d && ln -sf '%s' $$$$d"%(target,srcpath))

def library(env,name,libs=(),skip=(),extra=(),skip_all=False,no_exports=False,pyname=None):
  if name in env['skip_libs']:
    return
  sources = []
  headers = []
  skip = tuple(skip)+tuple(env['skip'])
  dir = Dir('.').srcnode().abspath
  has_module_cpp = False
  candidates = list(files(dir,skip)) if not skip_all else []
  for f in candidates + list(extra):
    if f.endswith('.cpp') or f.endswith('.cc') or f.endswith('.c'):
      if f.endswith('module.cpp'):
        has_module_cpp = True
      sources.append(f)
    elif f.endswith('.h'):
      headers.append(f)
  if not sources and not headers:
    print 'Warning: library %s has no input source files'%name
  if env.get('need_qt',0) and env['use_qt']: # Qt gets confused if we only set options on the builder
    env = env.Clone()
    force_external('qt')
    env.Append(CXXFLAGS=externals['qt']['cxxflags'])

  # Install headers
  for h in headers:
    install_or_link(env,os.path.join(env['prefix_include'],Dir('.').srcnode().path,os.path.dirname(h)),h)

  # Tell the compiler which library we're building
  env.Append(CPPDEFINES=['BUILDING_'+name])

  sources = objects(env,sources)
  env = link_flags(env)
  env = copy_files(env)

  libpath = '#'+os.path.join(env['variant_build'],'lib')
  path = os.path.join(libpath,name)
  env.Append(LIBS=libs)
  if env['shared']:
    linkflags = env['LINKFLAGS']
    if darwin:
      install_name = quote('%s/${SHLIBPREFIX}%s${SHLIBSUFFIX}'%(Dir(env.subst(env['prefix_lib'])).abspath,name))
      linkflags = '-install_name %s '%install_name+linkflags
    # On Windows, this will create two files: a .lib (for other builds), and a .dll for the runtime.
    lib = env.SharedLibrary(path,source=sources,LINKFLAGS=linkflags)
  else:
    lib = env.StaticLibrary(path,source=sources)
  env.Depends('.',lib)
  # Install dlls in bin, lib and exp in lib
  if windows:
    installed = []
    for l in lib:
      if l.name[-4:] in ['.dll','.pyd']:
        installed.extend(env.Install(env['prefix_bin'],l))
      elif not no_exports:
        installed.extend(env.Install(env['prefix_lib'],l))
    lib = installed
    all_libs.append(lib)
    if env['use_python'] and has_module_cpp:
      python_libs.append(lib)
  else:
    for l in lib:
      install_or_link(env, env['prefix_lib'], l)
    if env['use_python']:
      if pyname is None:
        pyname = name
      module = os.path.join('#'+Dir('.').srcnode().path,pyname)
      module = python_env.LoadableModule(module,source=[],LIBS=name,LIBPATH=[libpath])
      python_env.Depends(module,lib) # scons doesn't always notice this (obvious) dependency

def config_header(env,name,extra=()):
  """Generate a configuration header in the current directory"""
  env = env.Clone()
  add_dependencies(env)

  # Collect flags
  flags = []
  if env['real']=='float':
    flags.append('GEODE_FLOAT')
  flags.append(('GEODE_THREAD_SAFE',int(env['thread_safe'])))
  if env['sse']:
    flags.append('GEODE_SSE')
  if windows:
    if not env['shared']:
      flags.append('GEODE_SINGLE_LIB')
    flags.extend(('_CRT_SECURE_NO_DEPRECATE','NOMINMAX','_USE_MATH_DEFINES'))
  for n,lib in externals.items():
    if env.get('need_'+n):
      flags.extend(lib['flags'])
  # Turn off boost::exceptions to avoid completely useless code bloat
  flags.append('BOOST_EXCEPTION_DISABLE')

  # Generate header
  lines = ['// Autogenerated configuration header','#pragma once']
  for flag in flags:
    assert isinstance(flag,(str,bytes,tuple)),'Flags must be string or (string,value), got %s'%flag
    n,v = flag if isinstance(flag,tuple) else (flag,None)
    lines.append('\n#ifndef %s\n#define %s%s\n#endif'%(n,n,('' if v is None else ' %s'%v)))
  if extra:
    lines.append('')
    lines.extend(extra)
  header, = env.Textfile(name,lines)
  env.Depends('.',header)
  install_or_link(env,os.path.join(env['prefix_include'],os.path.dirname(header.srcnode().path)),header)

# Build a program
def program(env,name,cpp=None):
  if env['skip_programs']:
    return
  if cpp is None:
    cpp = name + '.cpp'
  env = link_flags(env)
  env = copy_files(env)
  files = objects(env,cpp)
  bin = env.Program('#'+os.path.join(env['variant_build'],'bin',name),files)
  env.Depends('.',bin)
  for b in bin:
    install_or_link(env, env['prefix_bin'], b)

# Install a (possibly directory) resource
def resource(env,path):
  # If we are installing, and we're adding a directory, add a dependency for each file found in its subtree
  node = dir_or_file(path).srcnode()
  if env['install'] and os.path.isdir(str(node)):
    def visitor(basedir, dirname, names):
      reldir = os.path.relpath(dirname, basedir)
      for name in names:
        fullname = os.path.join(dirname, name)
        install_or_link(env, os.path.join(env['prefix_share'], reldir), fullname)
    basedir = str(Dir('.').srcnode())
    os.path.walk(str(node), visitor, basedir)
  else:
    # if we're just making links, a single link is enough even if it's a directory
    install_or_link(env, env['prefix_share'], node)

# Configure latex
def configure_latex():
  def check(context):
    context.Message('checking for latex: ')
    r = context.TryBuild(latex_env.PDF,text=r'\documentclass{book}\begin{document}\end{document}',extension='.tex')
    context.Result(r)
    return r
  conf = latex_env.Configure(custom_tests={'Check':check})
  if not conf.Check():
    latex_env['use_latex'] = 0
  conf.Finish()
latex_configured = False

# Turn a latex document into a pdf
def latex(name):
  if latex_env['use_latex']:
    global latex_configured
    if not latex_configured:
      latex_configured = True
      configure_latex()
    if latex_env['use_latex']:
      pdf = os.path.join('#'+Dir('.').srcnode().path,name+'.pdf')
      latex_env.PDF(pdf,name+'.tex')

# Automatic python configuration
def configure_python(env,python):
  pattern = re.compile(r'^\s+',flags=re.MULTILINE)
  data = subprocess.Popen([env['python'],'-c',pattern.sub('','''
    import numpy
    import distutils.sysconfig as sc
    get = sc.get_config_var
    def p(s): print "'%s'"%s
    p(sc.get_python_inc())
    p(numpy.get_include())
    p(get('LIBDIR'))
    p(get('LDLIBRARY'))
    p(get('PYTHONFRAMEWORKPREFIX'))
    p(get('VERSION'))
    p(get('prefix'))
    ''')],cwd=os.path.abspath(os.sep),stdout=subprocess.PIPE).communicate()[0]
  include,nmpy,libpath,lib,frameworkpath,version,prefix = [s.strip()[1:-1] for s in data.strip().split('\n')]
  assert include,nmpy
  python['cpppath'] = [include] if os.path.exists(os.path.join(include,'numpy')) else [include,nmpy]
  if darwin:
    python_config = 'python%s-config' % version
    python['linkflags'] = subprocess.check_output([python_config,'--ldflags'])
  elif windows:
    python['libpath'] = [prefix,os.path.join(prefix,'libs')]
    python['libs'] = ['python%s'%version]
  else:
    assert libpath and lib and libpath!='None' and lib!='None'
    python['libpath'] = [libpath]
    python['libs'] = [lib]
  return 1

# Automatic MPI configuration
def configure_mpi(env,mpi):
  # Find the right mpicc
  if env['mpicc']=='<detect>':
    if windows:
      env.Replace(mpicc='mpicc')
    else:
      mpicc_options = ['mpicc','openmpicc']
      for mpicc in mpicc_options:
        if subprocess.Popen(['which',mpicc], stdout=subprocess.PIPE).communicate()[0]:
          env.Replace(mpicc=mpicc)
          break
      else:
        if verbose:
          print 'disabling mpi: mpicc not found'
        return 0

  # Configure MPI if it exists
  try:
    if env['mpicc'] and not mpi['cpppath']:
      # Find mpi.h
      mpi_include_options = ['/opt/local/include/openmpi','/usr/local/include/openmpi','/opt/local/include/mpi','/usr/local/include/mpi']
      for dir in mpi_include_options:
        if os.path.exists(os.path.join(dir,'mpi.h')):
          mpi['cpppath'] = dir
          break
    if env['mpicc'] and not (mpi['cxxflags'] or mpi['linkflags'] or mpi['libs']):
      for flags,stage in ('linkflags','link'),('cxxflags','compile'):
        mpi[flags] = ' '+subprocess.Popen([env['mpicc'],'--showme:%s'%stage],stdout=subprocess.PIPE).communicate()[0].strip()
      all_flags = mpi['linkflags'].strip().split()
      flags = []
      for f in all_flags:
        if f.startswith('-l'):
          mpi['libs'].append(f[2:])
        else:
          flags.append(f)
      mpi['linkflags'] = ' '+' '.join(flags)
  except OSError,e:
    if verbose:
      print 'disabling mpi: %s'%e
    return 0
  return 1

# Predefined external libraries
external(env,'python',default=1,flags=['GEODE_PYTHON'],configure=configure_python)
external(env,'boost',default=1,hide=1,headers=['boost/version.hpp'])
external(env,'boost_link',requires=['boost'],libs=['boost_iostreams$boost_lib_suffix',
  'boost_filesystem$boost_lib_suffix','boost_system$boost_lib_suffix','z','bz2'],hide=1,headers=())
external(env,'mpi',flags=['GEODE_MPI'],configure=configure_mpi)

# BLAS is tricky.  We define separate externals for openblas, atlas, and mkl,
# then a unified external which picks one of them.
def blas_variants():
  body = ['  cblas_dscal(0,1,0,1);']
  external(env,'atlas',libs=['cblas','lapack','atlas'],headers=['cblas.h'],body=body)
  external(env,'openblas',libs=['lapack','blas'],headers=['cblas.h'],body=body)
  if darwin:  external(env,'accelerate',frameworks=['Accelerate'],headers=['Accelerate/Accelerate.h'],body=body)
  if windows: external(env,'mkl',flags=['GEODE_MKL'],headers=['mkl_cblas.h'],body=body,libs='mkl_intel_lp64 mkl_intel_thread mkl_core mkl_mc iomp5 mkl_lapack'.split())
  else:       external(env,'mkl',flags=['GEODE_MKL'],headers=['mkl_cblas.h'],body=body,linkflags='-Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_mc -liomp5 -lmkl_lapack -Wl,--end-group -fopenmp -pthread')
blas_variants()
def configure_blas(env,blas):
  kinds = ['accelerate']*darwin+['openblas','atlas','mkl']
  for kind in kinds:
    force_external(kind)
    if env['use_'+kind]:
      print 'configuring blas: using %s'%kind
      blas['requires'] = [kind]
      return 1
  print>>sys.stderr, "disabling blas: can't find any variant, tried %s, and %s"%(', '.join(kinds[:-1]),kinds[-1])
  return 0
external(env,'blas',default=1,flags=['GEODE_BLAS'],configure=configure_blas)

# Descend into a child SConscript
def child(env,dir):
  # Descent into a subdirectory
  if dir in env['skip']:
    return
  path = Dir(dir).path
  variant = env['variant_build']
  if not path.startswith(variant):
    path = os.path.join(variant,path)
  env.SConscript('#'+os.path.join(path,'SConscript'),exports='env')

# Descend into all child SConscripts in order of priority
def children(env,skip=()):
  # Directories that define externals used by other directories must come first.
  # Therefore, we sort children with .priority files first in increase order of priority.
  def priority(dir):
    try:
      return float(open(File(os.path.join(dir,'.priority')).srcnode().abspath).read())
    except IOError:
      return 1e10
  base = Dir('.').srcnode().abspath+'/'
  dirs = [s[len(base):-11] for s in glob.glob(base+'*/SConscript')]
  for dir in sorted(dirs,key=priority):
    if dir not in skip and os.path.exists(File(os.path.join(dir,'SConscript')).srcnode().abspath):
      child(env,dir)

# Build everything
Export('''child children options external has_external library objects program latex
          clang posix darwin windows resource install_or_link config_header''')
if os.path.exists(File('#SConscript').abspath):
  child(env,'.')
else:
  children(env)

# If we're in msvc mode, build a toplevel solution
if windows and 0:
  env.MSVSSolution('#windows/other'+env['MSVSPROJECTSUFFIX'],projects=projects,variant=env['type'].capitalize())

# On Windows, distinct python extension modules can't share symbols.  Therefore, we
# build a single large extension module with links to all the dlls.
if windows and env['use_python']:
  if env['shared']:
    raise RuntimeError('Separate shared libraries do not work on windows.  Switch to shared=0.')

  # Autogenerate a toplevel module initialization routine calling all child initialization routines
  def make_modules(env,target,source):
    libs = str(source[0]).split()
    open(target[0].path,'w').write('''\
// Autogenerated by SConstruct: DO NOT EDIT
#define GEODE_PYTHON
#define GEODE_SINGLE_LIB
#include <geode/python/module.h>
#define SUB(name) void geode_init_helper_##name(); geode_init_helper_##name();
GEODE_PYTHON_MODULE(geode_all) {
  %s
}
'''%'\n  '.join('SUB(%s)'% (name if name != 'geode' else 'geode_wrap') for name in libs))
  modules, = env.Command(os.path.join(env['variant_build'],'modules.cpp'),
                         Value(' '.join(lib[0].name[:-4] for lib in python_libs)),[make_modules])

  # Build geode_all.pyd
  env = env.Clone(SHLIBSUFFIX='.pyd',shared=1)
  for use in all_uses:
    env[use] = 1
  geode_all = library(env,os.path.join(env['variant_build'],'geode_all'),
                      [lib[0].name[:-4] for lib in all_libs],
                      extra=(modules.path,),skip_all=True,no_exports=True)
  env.Alias('py',geode_all)

