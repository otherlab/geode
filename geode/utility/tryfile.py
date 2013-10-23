"""geode binary file format (.try)

A .try file holds packed, hierarchical binary data.  It consists of a
header, a tree of atoms, and a data section.  Each atom has a name, a
type, and some data, which is either more atoms or binary data in
various formats depending on the type.  Atoms can either include their
data inline or specify an offset into the data section.

All data is little-endian.  Integers are stored via the variable size
uint type.  The low 7 bits of each byte in a uint are part of the
number, and the high bit indicates whether another byte follows.
Thus integers from 0 to 2^7-1 occupy 1 byte, integers from 2^7 to
2^14-1 occupy 2 bytes, etc.  Strings are stored as

    uint size
    char string[size]

A .try file contains

    char signature[4] = chr(3)+'Try' # 'Try' as a string
    uint header_size                 # size of rest of header
    uint version = 2                 # for now
    uint tree_size                   # size of the tree section
    uint data_size                   # size of the data section
    int32 tree_crc                   # crc32 of the tree section
    atom tree                        # tree of atoms
    ... data                         # data section

An atom is

    string name
    string type
    uint version
    uint flags
    if leaf:
        uint data_size      # size of compressed data in data section
        if crc:
            int32 data_crc  # crc32 of the compressed data
    else:
        uint size
        atom children[size]

The possible flags are:

    IsLeaf = 1    # if false, data is an inline sequence of atoms
    Compressed = 2 # whether the data is compressed (leaf only)
    CRC = 4        # whether the data has a crc32 checksum (leaf only)
    
The data section contains the concatenation of the data from all leaves
in order of their appearance in the tree section.  If Compressed is set,
the data is compressed using zlib.  Note that in this case, data_size is
the size of the compressed data.

The initial set of atoms types and their data are

    dict: nonleaf
        Contents represent a dictionary from name to contents.
        The order of the child atoms is not significant.

    array: leaf
        Multidimensional numpy-like array.  The data is

            uint dtype
            uint rank
            uint shape[rank]
            ... array ...

        The available dtypes are:

            bool = 0
            char = 1
            int8 = 2
            uint8 = 3
            int16 = 4
            uint16 = 5
            int32 = 6
            uint32 = 7
            int64 = 8
            uint64 = 9
            float32 = 12
            float64 = 13

        Scalars can be represented as rank zero arrays.

The current implementation only supports reading and writing entire
files at once (i.e., no sparse or incremental support).  These would
be fairly straightforward to add, but both require seeks (either to
skip data in the read case or fill in data size information in the
write case).

To accomodate multiple frames, a future version may allow several
.try files to be concatenated with no extra header information added.

"""

__all__ = 'read write pack unpack Atom'.split()

import sys
import zlib
import struct
from cStringIO import StringIO

signature = '\003TRY'
current_version = 2

nonleaf_makers = {}
nonleaf_parsers = {}
leaf_makers = {}
leaf_parsers = {}

def register_nonleaf(typename,type,maker,parser,version=0):
  """Register a nonleaf atom type.

  maker(value) should return a series of (name,value) pairs, and
  parser(pairs,version) should convert the given (name,value) pairs
  into a value.
  
  """
  nonleaf_makers[type] = typename,version,maker 
  nonleaf_parsers[typename] = type,parser

def register_leaf(typename,type,maker,parser,version=0):
  """Register a leaf atom type.

  maker(value) should return a string containing the binary contents
  of value, and parser(data,version) should unpack the binary string
  data into a value.

  """
  leaf_makers[type] = typename,version,maker
  leaf_parsers[typename] = type,parser

def register_subtype(typename,type):
  if typename in leaf_parsers:
    leaf_makers[type] = leaf_makers[leaf_parsers[typename][0]]
  elif typename in nonleaf_parsers:
    nonleaf_makers[type] = nonleaf_makers[nonleaf_parsers[typename][0]]
  else:
    raise ValueError("atom type '%s' is not registered"%typename)

already_warned = set()

def warn_unknown(type):
  if type not in already_warned:
    if type in nonleaf_parsers:
      raise IOError("Leaf atom has nonleaf type '%s'"%type)
    elif type in leaf_parsers:
      raise IOError("Nonleaf atom has leaf type '%s'"%type)
    else:
      print>>sys.stderr, "warning: unknown atom type '%s'"%type
      already_warned.add(type)

def read_uint(file):
  """Read a uint from a file."""
  result = 0
  shift = 0
  while True:
    try:
      byte = ord(file.read(1))
    except TypeError:
      raise EOFError
    if byte&128:
      result |= (byte&127)<<shift
      shift += 7
    else:
      return result|(byte<<shift)

def uint_to_str(i):
  """Convert a uint to a string."""
  s = ''
  while True:
    if i>127:
      s += chr(i&127|128)
      i >>= 7
    else:
      return s+chr(i)

def read_string(file):
  return file.read(read_uint(file))

def string_to_str(s):
  return uint_to_str(len(s))+s

def read_crc(file):
  """Read a crc32 from a file."""
  return struct.unpack('<i',file.read(4))[0]

def crc_to_str(crc):
  """Convert a crc32 to a string."""
  return struct.pack('<I',crc%2**32)

IsLeaf = 1
Compressed = 2
CRC = 4
FlagMask = 7

class Atom(object):
  __slots__ = ['name','type','version','flags','data_size','data_crc']

  def to_str(self):
    return ''.join([string_to_str(self.name),string_to_str(self.type),uint_to_str(self.version),uint_to_str(self.flags)])

class Leaf(Atom):
  __slots__ = ['data']

  def to_str(self):
    return Atom.to_str(self)+uint_to_str(self.data_size)+crc_to_str(self.data_crc)

  def parse(self,file):
    data = file.read(self.data_size)
    if self.flags&CRC and (self.data_crc-zlib.crc32(data))%2**32:
      raise IOError('data crc32 mismatch: expected %d, got %d'%(self.data_crc,zlib.crc32(data)))
    if self.flags&Compressed:
      data = zlib.decompress(data)
    def unknown(version,data):
      warn_unknown(self.type)
      self.data = data
      return self
    _,parser = leaf_parsers.get(self.type,unknown)
    return parser(data,self.version)

  def write_data(self,file):
    file.write(self.data)

class Nonleaf(Atom):
  __slots__ = ['children']

  def to_str(self):
    return ''.join([Atom.to_str(self),uint_to_str(len(self.children))]+[c.to_str() for c in self.children])

  def parse(self,file):
    children = [(c.name,c.parse(file)) for c in self.children]
    def unknown(version,children):
      warn_unknown(self.type)
      self.children = children
      return self
    _,parser = nonleaf_parsers.get(self.type,unknown)
    return parser(children,self.version)

  def write_data(self,file):
    for c in self.children:
      c.write_data(file)

def read_atom(file):
  name = read_string(file)
  type = read_string(file)
  version = read_uint(file)
  flags = read_uint(file)
  if flags&~FlagMask:
    raise IOError("unknown flags %d"%(flags&~FlagMask))
  if flags&IsLeaf:
    atom = Leaf()
    atom.data_size = read_uint(file)
    if flags&CRC:
      atom.data_crc = read_crc(file)
  else:
    atom = Nonleaf()
    atom.children = [read_atom(file) for _ in range(read_uint(file))]
    atom.data_size = sum(a.data_size for a in atom.children)
  atom.name = name
  atom.type = type
  atom.version = version
  atom.flags = flags
  return atom

def make_atom(name,value):
  t = type(value)
  if t in leaf_makers:
    atom = Leaf()
    atom.type,atom.version,maker = leaf_makers[t]
    atom.data = zlib.compress(maker(value))
    atom.data_size = len(atom.data)
    atom.data_crc = zlib.crc32(atom.data)
    atom.flags = IsLeaf|Compressed|CRC
  elif t in nonleaf_makers:
    atom = Nonleaf()
    atom.type,atom.version,maker = nonleaf_makers[t]
    atom.children = [make_atom(*p) for p in maker(value)]
    atom.data_size = sum(c.data_size for c in atom.children)
    atom.flags = 0
  else:
    raise TypeError("can't convert unregistered type '%s' to atom"%t.__name__)
  atom.name = name
  return atom

def read_stream(file):
  '''Read a .try file from an open stream.'''
  if file.read(4)!=signature:
    raise IOError('bad signature')
  header_size = read_uint(file)
  header_start = file.tell()
  version = read_uint(file)
  if not 1<=version<=2:
    raise IOError('unknown version %d'%version)
  tree_size = read_uint(file)
  data_size = read_uint(file)
  if version>1:
    tree_crc = read_crc(file)
  if header_size<file.tell()-header_start:
    raise IOError('header_size smaller than header')
  tree_start = header_start+header_size
  file.seek(tree_start) # Skip over the rest of the header

  # Read atom tree.  This reads and parses the entire atom hierarchy, but does not parse the leaf data.
  tree = file.read(tree_size)
  tree_file = StringIO(tree)
  try:
    atom = read_atom(tree_file)
  except EOFError:
    raise IOError('unexpected end of tree section, size %d is too small'%tree_size)
  if version>1 and (tree_crc-zlib.crc32(tree))%2**32:
    raise IOError('tree crc32 mismatch: expected %d, got %d'%(tree_crc,zlib.crc32(tree)))
  if tree_file.tell()!=tree_size:
    raise IOError('expected tree size %d, got %d'%(tree_size,tree_file.tell()))
  data_start=file.tell()

  # Read data section by traversing the atom tree.
  result = atom.parse(file)
  data_end = file.tell()
  if data_end-data_start!=data_size:
    raise IOError('expected data size %d, got %d'%(data_size,data_end-data_start))
  return result

def write_stream(file,value):
  '''Write a .try file to an open stream.'''
  # Build atom tree in memory 
  atom = make_atom('',value)
  tree = atom.to_str()

  # Write header
  file.write(signature)
  tree_size = len(tree)
  data_size = atom.data_size
  tree_crc = zlib.crc32(tree)
  header = ''.join(uint_to_str(i) for i in (current_version,tree_size,data_size))+crc_to_str(tree_crc)
  file.write(string_to_str(header))

  # Write tree
  file.write(tree)

  # Write data
  atom.write_data(file)

def read(filename):
  '''Read the contents of a .try file in its entirety.

  Return (name,data), where data is the parsed contents of the
  toplevel tree atom.  The mapping from atom data to python types
  is dict to dict, array to array, as one would expect.  Unknown
  atom types are parsed into Atom.'''
  return read_stream(open(filename,'rb'))

def write(filename,value):
  '''Write a new .try file in its entirety.

  Data must be a nested tree of dictionaries with scalars or numpy
  arrays as leaves.'''
  write_stream(open(filename,'wb'),value)

def unpack(buffer):
  '''Unpack a string in .try format into data.'''
  return read_stream(StringIO(buffer))

def pack(value):
  '''Pack data into a .try format string.'''
  file = StringIO()
  write_stream(file,value)
  return file.getvalue()

### Dict

def parse_dict(pairs,version):
  return dict(pairs)

register_nonleaf('dict',dict,dict.iteritems,parse_dict)

### Array

import numpy

# Start dtype map off with platform independent dtypes only
int_to_dtype = map(numpy.dtype,'bool int8 uint8 int16 uint16 int32 uint32 int64 uint64 float32 float64'.split())
dtype_num_to_int = dict((d.num,i) for i,d in enumerate(int_to_dtype))

def make_array(a):
  a = numpy.asarray(a)
  try:
    dtype = dtype_num_to_int[a.dtype.num]
  except KeyError:
    # dtype isn't correctly hashable, so do linear search for matching dtype
    for i,d in enumerate(int_to_dtype):
      if a.dtype==d:
        dtype = dtype_num_to_int[a.dtype.num]=dtype_num_to_int[d.num]
        break
    else:
      raise TypeError("unregistered dtype '%s'"%a.dtype)
  # Convert numpy array to little endian buffer, flipping endianness if necessary
  return ''.join([uint_to_str(i) for i in (dtype,len(a.shape))+a.shape]+[a.astype(a.dtype.newbyteorder('<')).tostring()])

def parse_array(data,version):
  file = StringIO(data)
  dtype = int_to_dtype[read_uint(file)]
  rank = read_uint(file)
  shape = [read_uint(file) for _ in xrange(rank)]
  # Convert little endian buffer to a numpy array, flipping endianness if necessary
  array = (numpy.frombuffer(data,dtype=dtype.newbyteorder('<'),offset=file.tell()).astype(dtype) if numpy.product(shape) else numpy.empty(0,dtype)).reshape(shape)
  return numpy.require(array,requirements='a')

register_leaf('array',numpy.ndarray,make_array,parse_array)
for t in int,bool,float,numpy.int32,numpy.int64,numpy.float32,numpy.float64:
  register_subtype('array',t)

### Str

register_leaf('str',str,str,lambda s,v:s)

### Tuple and List

def tuple_maker(x):
  return ((str(i),y) for i,y in enumerate(x))

def parse_tuple(pairs,version):
  x = []
  for i,(n,y) in enumerate(pairs):
    assert i==int(n)
    x.append(y) 
  return tuple(x)

register_nonleaf('list',list,tuple_maker,parse_tuple)
register_nonleaf('tuple',tuple,tuple_maker,parse_tuple)
