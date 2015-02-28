// Mesh file I/O

#include <geode/mesh/io.h>
#include <geode/mesh/PolygonSoup.h>
#include <geode/array/view.h>
#include <geode/geometry/Triangle3d.h>
#include <geode/python/cast.h>
#include <geode/python/wrap.h>
#include <geode/utility/endian.h>
#include <geode/utility/function.h>
#include <geode/utility/path.h>
#include <errno.h>
namespace geode {

typedef real T;
typedef Vector<T,3> TV;
typedef Vector<T,2> TV2;
using std::cout;
using std::endl;

namespace {
struct File {
  FILE* const f;

  File(const string& filename, const char* mode)
    : f(fopen(filename.c_str(),mode)) {
    if (!f)
      throw IOError(format("can't open '%s' for %s: %s",filename,
        mode[0]=='r' ? "reading" : mode[0]=='w' ? "writing" : mode[0]=='a' ? "append" : format("mode %s",mode),
        strerror(errno)));
  }

  File(const File&) = delete;
  void operator=(const File&) = delete;

  ~File() {
    fclose(f);
  }

  operator FILE*() const {
    return f;
  }
};
}

// Determine whether a file is probably binary or ascii
static bool is_binary(const string& filename) {
  File f(filename,"rb");
  char data[512];
  const int n = int(fread(data,1,sizeof(data),f));
  for (int i=0;i<n;i++)
    if (!isascii(data[i]))
      return true;
  return false;
}

static const char* skip_white(const char* p) {
  while (isspace(*p))
    p++;
  return p;
}

// Can't use a simple struct since StlTri has bad alignment
struct StlTriData {
  Vector<float,3> n;
  Vector<float,3> x[3];
};
struct StlTri {
  char d[sizeof(StlTriData)];
  uint16_t c;
};
static_assert(sizeof(StlTri)==12*4+2,"");

// See http://en.wikipedia.org/wiki/STL_file for details
static Tuple<Ref<TriangleSoup>,Array<TV>> read_stl(const string& filename) {
  if (is_binary(filename)) {
    File f(filename,"rb");

    // Skip header
    char header[80];
    const auto nh = fread(header,1,sizeof(header),f);
    if (nh < sizeof(header))
      throw IOError(format("invalid binary stl '%s': %s",filename,
        ferror(f) ? format("read failed: %s",strerror(errno)) : "incomplete header"));

    // Read count
    uint32_t count;
    const auto nc = fread(&count,sizeof(count),1,f);
    if (nc < 1)
      throw IOError(format("invalid binary stl '%s': failed to read count",filename));
    if (count > (1u<<31)/3-1)
      throw IOError(format("binary stl has too many triangles: %u > 2^31/3-1",count));

    // Read triangles
    Array<StlTri> data(count,uninit);
    const auto nt = fread(data.data(),sizeof(StlTri),count,f);
    if (nt < count)
      throw IOError(format("invalid binary stl '%s': failed to read triangles",filename));

    // Deduplicate
    Hashtable<Vector<float,3>,int> id;
    Array<Vector<int,3>> tris;
    Array<TV> X;
    for (auto& t : data) {
      StlTriData d;
      memcpy(&d,t.d,sizeof(d));
      Vector<int,3> tri;
      for (int a=0;a<3;a++) {
        const int i = id.get_or_insert(d.x[a],X.size());
        if (i == X.size())
          X.append(TV(d.x[a]));
        tri[a] = i;
      }
      tris.append(tri);
    }
    return tuple(new_<TriangleSoup>(tris,X.size()),X);
  } else { // ASCII
    File f(filename,"r");

    // Prepare for deduplication
    Hashtable<Vector<double,3>,int> id;
    Array<Vector<int,3>> tris;
    Array<TV> X;

    // Read
    int ns = 0, nl = 0;
    char line[1024];
    for (;;) {
      nl++;
      if (!fgets(line,sizeof(line),f)) {
        if (!ns)
          throw IOError(format("invalid ascii stl %s: no solids found",filename));
        break;
      }
      {
        const char* p = skip_white(line);
        if (!*p) continue;
        if (strncmp(p,"solid ",6))
          throw IOError(format("invalid ascii stl %s:%d: expected 'solid ', got: %s",filename,nl,repr(p)));
      }
      for (;;) {
        nl++;
        if (!fgets(line,sizeof(line),f))
          throw IOError(format("invalid ascii stl %s:%d: file ended inside solid",filename,nl));
        {
          const char* p = skip_white(line);
          if (!*p) continue;
          if (!strncmp(p,"endsolid",8))
            break;
          if (strncmp(p,"facet normal ",13))
            throw IOError(format("invalid ascii stl %s:%d: expected 'endsolid' or 'facet normal ', got: %s",
              filename,nl,repr(p)));
        }
        static const int len[6] = {10,7,7,7,7,8};
        static const char* expect[6] = {"outer loop","vertex ","vertex ","vertex ","endloop","endfacet"};
        Vector<int,3> tri;
        for (int a=0;a<6;a++) {
          nl++;
          if (!fgets(line,sizeof(line),f))
            throw IOError(format("invalid ascii stl %s:%d: file ended inside facet",filename,nl));
          const char* p = skip_white(line);
          if (!*p) { a--; continue; }
          if (strncmp(p,expect[a],len[a]))
            throw IOError(format("invalid ascii stl %s:%d: expected '%s', got: %s",filename,nl,expect[a],repr(p)));
          if (1<=a && a<4) {
            Vector<double,3> x;
            if (sscanf(p+7,"%lg %lg %lg",&x.x,&x.y,&x.z) != 3)
              throw IOError(format("invalid ascii stl %s:%d: invalid vertex line: %s",filename,nl,repr(p)));
            const int i = id.get_or_insert(x,X.size());
            if (i == X.size()) {
              if (X.size()==numeric_limits<int>::max())
                throw IOError(format("ascii stl %s has too many vertices, our limit is 2^31-1",filename));
              X.append(x);
            }
            tri[a-1] = i;
          }
        }
        if (tris.size()==numeric_limits<int>::max())
          throw IOError(format("ascii stl %s has too many vertices, our limit is 2^31-1",filename));
        tris.append(tri);
      }
      ns++;
    }
    return tuple(new_<TriangleSoup>(tris,X.size()),X);
  }
}

static void write_stl(const string& filename, RawArray<const Vector<int,3>> tris, RawArray<const TV> X) {
  // We unconditionally write .stl files in binary.  Text formats for large data are silly.
  File f(filename,"wb");
  fprintf(f,"%-79s\n","Binary STL triangle mesh: http://en.wikipedia.org/wiki/STL_file");
  const uint32_t count = tris.size();
  fwrite(&count,sizeof(count),1,f);
  for (const auto nodes : tris) {
    StlTriData d;
    d.n = to_little_endian(Vector<float,3>(normal(X[nodes[0]],X[nodes[1]],X[nodes[2]])));
    for (int i=0;i<3;i++)
      d.x[i] = to_little_endian(Vector<float,3>(X[nodes[i]]));
    StlTri t;
    memcpy(t.d,&d,sizeof(d));
    t.c = 0;
    fwrite(&t,sizeof(t),1,f);
  }
}

static const char* white = " \t\v\f\r\n";

#ifdef _WIN32
static char* strtok_r(char* str, const char* delim, char** saveptr) {
  return strtok_s(str, delim, saveptr);
}
#endif

static Tuple<Ref<PolygonSoup>,Array<TV>> read_obj(const string& filename) {
  File f(filename,"r");

  // Parse file
  Array<TV> X, normals;
  Array<TV2> texcoords;
  Array<int> counts, vertices;
  char orig[1024], line[1024];
  int nl = 0;
  while (fgets(orig,sizeof(orig),f)) {
    nl++;
    strcpy(line,orig);
    char* save;
    const char* cmd = strtok_r(line,white,&save);
    if (!cmd || cmd[0] == '#')
      continue;
    else if (cmd[0]=='v' && (!cmd[1] || ((cmd[1]=='n' || cmd[1]=='t') && !cmd[2]))) { // cmd = v, vn, or vt
      int n = 0;
      double x[4];
      for (int i=0;i<4;i++)
        if (const char* q = strtok_r(0,white,&save)) {
          char* end;
          x[n++] = strtod(q,&end);
          if (*end)
            throw IOError(format("invalid obj file %s:%d: bad %s line: %s",filename,nl,cmd,repr(orig)));
        }
      const int ne = !cmd[1] ? 3 : cmd[1]=='n' ? 3 : /*cmd[1]=='t'*/ 2;
      if (n != ne)
        throw IOError(format("invalid obj file %s:%d: %s expected 3 floats, got %s",filename,nl,cmd,ne,repr(orig)));
      if (!cmd[1]) { // v
        if (X.size()==numeric_limits<int>::max())
          throw IOError(format("unsupported obj file %s: too many vertices (our limit is 2^31-1)",filename));
        X.append(TV(x[0],x[1],x[2]));
      } else if (cmd[1]=='n') { // vn
        if (normals.size()==numeric_limits<int>::max())
          throw IOError(format("unsupported obj file %s: too many normals (our limit is 2^31-1)",filename));
        normals.append(TV(x[0],x[1],x[2]));
      } else { // vt
        if (texcoords.size()==numeric_limits<int>::max())
          throw IOError(format("unsupported obj file %s: too many texcoords (our limit is 2^31-1)",filename));
        texcoords.append(TV2(x[0],x[1]));
      }
    } else if (cmd[0]=='f' && !cmd[1]) { // cmd = f
      int n = 0;
      while (const char* q = strtok_r(0,white,&save)) {
        n++;
        char* end;
        const long v = strtol(q,&end,0);
        if (*end && *end != '/')
          throw IOError(format("invalid obj file %s:%d: f expected ints, got %s",filename,nl,repr(orig)));
        // TODO: Don't skip face normal or face texcoord information
        if (long(unsigned(int(v))) != v)
          throw IOError(format("unsupported obj file %s:%d: f got invalid vertex id %ld",filename,nl,v));
        vertices.append(int(v));
      }
      if (n < 3)
        throw IOError(format("invalid obj file %s:%d: f got fewer than 3 vertices",filename,nl));
      counts.append(n);
    } else if (strcmp(cmd,"usemtl") || strcmp(cmd,"usemat") || strcmp(cmd,"mtllib")) {
      // TODO: Don't skip these fields?
    } else
      throw IOError(format("invalid obj file %s:%d: invalid command %s",filename,nl,repr(cmd)));
  }

  // Adjust vertices and check consistency
  vertices -= 1;
  for (const int v : vertices)
    if (!X.valid(v))
      throw IOError(format("invalid obj file %s: face vertex %d out of valid range [1,%d]",filename,v+1,X.size()));
  if (normals.size() && normals.size() != X.size())
    throw IOError(format("invalid obj file %s: %d vertices != %d normals",filename,X.size(),normals.size()));
  if (texcoords.size() && texcoords.size() != X.size())
    throw IOError(format("invalid obj file %s: %d vertices != %d texcoords",filename,X.size(),texcoords.size()));

  // TODO: Don't discard normal and texcoord information
  return tuple(new_<PolygonSoup>(counts,vertices,X.size()),X);
}

static void write_obj_helper(File& f, RawArray<const TV> X) {
  // Write format
  fputs("# Simple obj file format: http://en.wikipedia.org/wiki/Wavefront_.obj_file\n"
        "#   # Vertex at coordinates (x,y,z):\n"
        "#   v x y z\n"
        "#   # Triangle [quad] with vertices a,b,c[,d]:\n"
        "#   f a b c [d]\n"
        "#   # Vertices are indexed starting from 1\n",f);

  // Write vertices
  for (const auto x : X)
    fprintf(f,"v %g %g %g\n",x.x,x.y,x.z);
}
static void write_obj(const string& filename, RawArray<const Vector<int,3>> tris, RawArray<const TV> X) {
  File f(filename,"wb");
  write_obj_helper(f,X);
  for (const auto t : tris)
    fprintf(f,"f %d %d %d\n",t.x+1,t.y+1,t.z+1);
}
static void write_obj(const string& filename, const PolygonSoup& soup, RawArray<const TV> X) {
  File f(filename,"wb");
  write_obj_helper(f,X);
  int offset = 0;
  for (const auto n : soup.counts) {
    fputc('f',f);
    for (int i=0;i<n;i++)
      fprintf(f," %d",soup.vertices[offset++]+1);
  }
}

namespace {
struct Line {
  int lineno;
  Array<char> line, split;
  Array<const char*> words;

  // TODO: Allow for longer lines
  Line()
    : lineno(0), line(1024), split(1024) {}

  bool read(File& f) {
    lineno++;
    if (!fgets(line.data(),line.size(),f))
      return false;
    strcpy(split.data(),line.data());
    char* p = split.data();
    char* save;
    words.clear();
    while (const char* w = strtok_r(p,white,&save)) {
      words.append(w);
      p = 0;
    }
    return true;
  }

  string repr() const {
    return geode::repr(line.data());
  }
};

struct PlyProp : public Object {
  GEODE_NEW_FRIEND
  const string name;
protected:
  PlyProp(const string& name)
    : name(name) {}
public:
  virtual void read_ascii(RawArray<const char*> words, int& i) = 0;
  virtual void read_binary_same_endian(FILE* f) = 0;
  virtual void read_binary_flip_endian(FILE* f) = 0;
  virtual string type() const = 0;
};

template<class T> static const char* ply_type_name();
#define PLY_TYPE_NAME(name,T) \
  template<> const char* ply_type_name<T>() { return #name; }
#define PLY_SIMPLE_TYPE(name) PLY_TYPE_NAME(name,name##_t)
PLY_SIMPLE_TYPE(int8)
PLY_SIMPLE_TYPE(uint8)
PLY_SIMPLE_TYPE(int16)
PLY_SIMPLE_TYPE(uint16)
PLY_SIMPLE_TYPE(int32)
PLY_SIMPLE_TYPE(uint32)
PLY_TYPE_NAME(float,float)
PLY_TYPE_NAME(double,double)

#define PLY_TYPE_NAMES(f) \
  f(char,int8_t) \
  f(uchar,uint8_t) \
  f(short,int16_t) \
  f(ushort,uint16_t) \
  f(int,int32_t) \
  f(uint,uint32_t) \
  f(float,float) \
  f(double,double) \
  f(int8,int8_t) \
  f(uint8,uint8_t) \
  f(int16,int16_t) \
  f(uint16,uint16_t) \
  f(int32,int32_t) \
  f(uint32,uint32_t) \
  f(float32,float) \
  f(float64,double)

template<class T> static inline typename enable_if<is_integral<T>,T>::type parse(const char* s) {
  char* end;
  const long long n = strtoll(s,&end,0);
  if (end[0])
    throw IOError(format("invalid %s value %s",ply_type_name<T>(),repr(s)));
  if ((long long)(T)n != n)
    throw IOError(format("out of range %s value %s",ply_type_name<T>(),repr(s)));
  return T(n);
}

template<class T> static inline typename enable_if<is_floating_point<T>,T>::type parse(const char* s) {
  char* end;
  const double x = strtod(s,&end);
  if (end[0])
    throw IOError(format("invalid %s value %s",ply_type_name<T>(),repr(s)));
  return x;
}

template<class T> struct PlyPropSingle : public PlyProp {
  GEODE_NEW_FRIEND
  Array<T> a;
protected:
  PlyPropSingle(const string& name, const int count)
    : PlyProp(name) {
    a.preallocate(count);
  }

  void read_ascii(RawArray<const char*> words, int& i) {
    if (i==words.size())
      throw IOError(format("incomplete element (no %s)",name));
    a.append_assuming_enough_space(parse<T>(words[i++]));
  }

  void read_binary_same_endian(FILE* f) {
    T x;
    if (fread(&x,sizeof(T),1,f) < 1)
      throw IOError(format("incomplete element (no %s)",name));
    a.append(x);
  }

  void read_binary_flip_endian(FILE* f) {
    T x;
    if (fread(&x,sizeof(T),1,f) < 1)
      throw IOError(format("incomplete element (no %s)",name));
    a.append(flip_endian(x));
  }

  string type() const {
    return ply_type_name<T>();
  }
};

template<class L,class T> struct PlyPropList : public PlyProp {
  GEODE_NEW_FRIEND
  static_assert(is_same<L,uint8_t>::value,"L must be uint8_t for now");
  Array<int> counts;
  Array<T> flat;
protected:
  PlyPropList(const string& name, const int n)
    : PlyProp(name) {
    counts.preallocate(n);
  }

  void read_ascii(RawArray<const char*> words, int& i) {
    if (i==words.size())
      throw IOError(format("incomplete element: no %s size",name));
    const auto n = parse<L>(words[i++]);
    counts.append_assuming_enough_space(n);
    flat.preallocate(flat.size()+n);
    for (int j=0;j<n;j++) {
      if (i==words.size())
        throw IOError(format("incomplete element: %s expected %d entries, got %d",name,n,j));
      flat.append_assuming_enough_space(parse<T>(words[i++]));
    }
  }

  void read_binary_same_endian(FILE* f) {
    L n;
    if (fread(&n,sizeof(L),1,f) < 1)
      throw IOError(format("incomplete element (no %s size)",name));
    counts.append_assuming_enough_space(n);
    const int offset = flat.size();
    flat.resize(flat.size()+n);
    if (fread(flat.data()+offset,sizeof(T),n,f) < n)
      throw IOError(format("incomplete element (incomplete %s list)",name));
  }

  void read_binary_flip_endian(FILE* f) {
    L n;
    if (fread(&n,sizeof(L),1,f) < 1)
      throw IOError(format("incomplete element (no %s size)",name));
    n = flip_endian(n);
    counts.append_assuming_enough_space(n);
    const int offset = flat.size();
    flat.resize(flat.size()+n);
    if (fread(flat.data()+offset,sizeof(T),n,f) < n)
      throw IOError(format("incomplete element (incomplete %s list)",name));
    for (int i=0;i<n;i++)
      flat[offset+i] = flip_endian(flat[offset+i]);
  }

  string type() const {
    return ply_type_name<T>();
  }
};

struct PlyElement : public Object {
  GEODE_NEW_FRIEND
  const string name;
  const int count;
  vector<Ref<PlyProp>> props;
  Hashtable<string,Ref<PlyProp>> prop_names;
protected:
  PlyElement(const string& name, const int count)
    : name(name)
    , count(count) {}
};
}

static Tuple<Ref<PolygonSoup>,Array<TV>> read_ply(const string& filename) {
  File f(filename,"rb");
  Line line;
  try {
    // Read magic string
    if (!line.read(f) || line.words.size()!=1 || strcmp(line.words[0],"ply")) {
      cout << "words = "<<line.words<<endl;
      throw IOError(format("expected magic string 'ply', got %s",repr(line)));
    }

    // Read rest of header
    int fmt = 0; // 1 for ascii, 2 for binary little endian, 3 for binary big endian
    vector<Ref<PlyElement>> elements;
    Hashtable<string,Ref<PlyElement>> element_names;
    for (;;) {
      if (!line.read(f))
        throw IOError("eof before end of header");
      const auto words = line.words.raw();
      if (!words.size() || !strcmp(words[0],"comment"))
        continue;
      else if (!strcmp(words[0],"format")) {
        if (fmt)
          throw IOError("duplicate format line");
        if (words.size() != 3)
          throw IOError(format("invalid format line %s",repr(line)));
        try {
          const double version = parse<double>(words[2]);
          if (version != 1)
            throw IOError("");
        } catch (const IOError&) {
          throw IOError(format("unsupported version %s",repr(words[2])));
        }
        if      (!strcmp(words[1],"ascii"))                fmt = 1;
        else if (!strcmp(words[1],"binary_little_endian")) fmt = 2;
        else if (!strcmp(words[1],"binary_big_endian"))    fmt = 3;
      } else if (!strcmp(words[0],"element")) {
        try {
          if (words.size() != 3)
            throw IOError("expected 'element <name> <count>'");
          const auto E = new_<PlyElement>(words[1],parse<int>(words[2]));
          if (!element_names.set(E->name,E))
            throw IOError(format("duplicate element name %s",repr(E->name)));
          elements.push_back(E);
        } catch (const IOError& e) {
          throw IOError(format("invalid element declaration %s: %s",repr(line),e.what()));
        }
      } else if (!strcmp(words[0],"property")) {
        if (!elements.size())
          throw IOError("property before element");
        PlyElement& E = elements.back();
        if (words.size() < 3)
          throw IOError("incomplete property declaration, expected 'property [list uchar] type name'");
        Ptr<PlyProp> prop;
        #define SINGLE_CASE(name,T) \
          else if (!strcmp(words[1],#name)) \
            prop = new_<PlyPropSingle<T>>(words[2],E.count);
        #define LIST_CASE(name,T) \
          else if (!strcmp(words[3],#name)) \
            prop = new_<PlyPropList<uint8_t,T>>(words[4],E.count);
        if (!strcmp(words[1],"list")) {
          if (words.size() != 5)
            throw IOError("invalid list property declaration, expected 'property list uchar type name'");
          if (strcmp(words[2],"uchar"))
            throw IOError(format("unsupported list property declaration, only uchar sizes are supported, got %s",
              repr(words[2])));
          PLY_TYPE_NAMES(LIST_CASE)
          else
            throw IOError(format("invalid list property type %s",repr(words[3])));
        } else {
          if (words.size() != 3)
            throw IOError("invalid single property declaration, expected 'property type name'");
          PLY_TYPE_NAMES(SINGLE_CASE)
          else
            throw IOError(format("invalid property type %s",repr(words[1])));
        }
        if (!E.prop_names.set(prop->name,ref(prop)))
          throw IOError(format("duplicate property name %s for element %s",repr(prop->name),repr(E.name)));
        E.props.push_back(ref(prop));
      } else if (!strcmp(words[0],"end_header"))
        break;
      else
        throw IOError(format("invalid header command %s",repr(words[0])));
    }
    if (!fmt)
      throw IOError("missing format declaration");

    #if GEODE_ENDIAN == GEODE_LITTLE_ENDIAN
      const int native = 2;
    #elif GEODE_ENDIAN == GEODE_BIG_ENDIAN
      const int native = 3;
    #endif

    // Read all elements
    if (fmt == 1) {
      for (const auto& E : elements) {
        for (const int i : range(E->count)) {
          if (!line.read(f))
            throw IOError(format("failed to read element %s, index %d: unexpected end of file",repr(E->name),i));
          int n = 0;
          for (const auto& prop : E->props) {
            try {
              prop->read_ascii(line.words,n);
            } catch (const IOError& e) {
              throw IOError(format("failed to read element %s, index %d, prop %s: %s",
                repr(E->name),i,repr(prop->name),e.what()));
            }
          }
          if (n != line.words.size())
            throw IOError(format("failed to read element %s, index %d: extra fields",repr(E->name),i));
        }
      }
    } else if (fmt == native) {
      for (const auto& E : elements) {
        for (const int i : range(E->count)) {
          for (const auto& prop : E->props) {
            try {
              prop->read_binary_same_endian(f);
            } catch (const IOError& e) {
              throw IOError(format("failed to read element %s, index %d, prop %s: %s",
                repr(E->name),i,repr(prop->name),e.what()));
            }
          }
        }
      }
    } else { // fmt == nonnative
      for (const auto& E : elements) {
        for (const int i : range(E->count)) {
          for (const auto& prop : E->props) {
            try {
              prop->read_binary_flip_endian(f);
            } catch (const IOError& e) {
              throw IOError(format("failed to read element %s, index %d, prop %s: %s",
                repr(E->name),i,repr(prop->name),e.what()));
            }
          }
        }
      }
    }

    // Pull out all the data we need
    // TODO: Don't discard all the rest of the data
    if (!element_names.contains("vertex"))
      throw IOError("missing vertex element");
    const auto vertex = element_names.get("vertex");
    Array<TV> X(vertex->count,uninit);
    for (const int i : range(3)) {
      const string c(1,"xyz"[i]);
      if (!vertex->prop_names.contains(c))
        throw IOError(format("vertex element missing property %s",c));
      const auto x_ = vertex->prop_names.get(c);
      if (const auto* x = dynamic_cast<PlyPropSingle<float>*>(&*x_)) {
        for (const int j : range(X.size()))
          X[j][i] = x->a[j];
      } else if (const auto& x = dynamic_cast<PlyPropSingle<double>*>(&*x_)) {
        for (const int j : range(X.size()))
          X[j][i] = x->a[j];
      } else
        throw IOError(format("vertex.%s has invalid type %s",c,x_->type()));
    }
    if (!element_names.contains("face"))
      throw IOError("missing face element");
    const auto face = element_names.get("face");
    if (!face->prop_names.contains("vertex_indices"))
      throw IOError("face element missing vertex_indices");
    const auto vertices = face->prop_names.get("vertex_indices");
    if (const auto* v = dynamic_cast<PlyPropList<uint8_t,int>*>(&*vertices))
      return tuple(new_<PolygonSoup>(v->counts,v->flat,X.size()),X);
    else
      throw IOError(format("face.vertex_indices has unsupported type %s",vertices->type()));
  } catch (const IOError& e) {
    throw IOError(format("invalid ply file %s:%d: %s",filename,line.lineno,e.what()));
  }
}

static void write_ply_helper(File& f, const int nfaces, RawArray<const TV> X) {
  fprintf(f,"ply\n"
            "format binary_little_endian 1.0\n"
            "comment Binary .ply file: http://en.wikipedia.org/wiki/PLY_(file_format)\n"
            "element vertex %d\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "element face %d\n"
            "property list uchar int vertex_indices\n"
            "end_header\n",X.size(),nfaces);
  for (const auto& x : X) {
    const auto y = to_little_endian(Vector<float,3>(x));
    fwrite(&y,sizeof(y),1,f);
  }
}

static void write_ply(const string& filename, RawArray<const Vector<int,3>> tris, RawArray<const TV> X) {
  File f(filename,"wb");
  write_ply_helper(f,tris.size(),X);
  for (const Vector<int,3>& t : tris) {
    const uint8_t n = 3;
    fwrite(&n,1,1,f);
    fwrite(&to_little_endian(t),sizeof(t),1,f);
  }
}

static void write_ply(const string& filename, const PolygonSoup& soup, RawArray<const TV> X) {
  File f(filename,"wb");
  write_ply_helper(f,soup.counts.size(),X);
  int offset = 0;
  for (const int c : soup.counts) {
    const uint8_t cb(c);
    if (int(c)!=c)
      throw IOError(format("write_ply: can't write face with %d > 255 vertices",c));
    fwrite(&cb,1,1,f);
    for (int i=0;i<c;i++) {
      const int v = to_little_endian(soup.vertices[offset++]);
      fwrite(&v,sizeof(v),1,f);
    }
  }
}

static void write_x3d_helper(const string& filename, const function<void(File&)>& write_topology, RawArray<const TV> X) {
  File f(filename,"wb");
  fputs("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<!DOCTYPE X3D PUBLIC \"ISO//Web3D//DTD X3D 3.1//EN\" \"http://www.web3d.org/specifications/x3d-3.1.dtd\">\n"
        "<X3D profile=\"Immersive\" version=\"3.1\" "
            "xsd:noNamespaceSchemaLocation=\"http://www.web3d.org/specifications/x3d-3.1.xsd\" "
            "xmlns:xsd=\"http://www.w3.org/2001/XMLSchema-instance\">\n"
        " <head>\n"
        "  <meta content=\".x3d format: http://en.wikipedia.org/wiki/X3d\" name=\"description\"/>\n"
        " </head>\n"
        " <Scene>\n"
        "  <Shape>\n"
        "   <IndexedFaceSet coordIndex=\"",f);
  write_topology(f);
  fputs("\" solid=\"false\">\n"
        "    <Coordinate point=\"",f);
  const auto flat = scalar_view(X);
  if (flat.size())
    fprintf(f,"%g",flat[0]);
  for (int i=1;i<flat.size();i++)
    fprintf(f," %g",flat[i]);
  fputs("\"/>\n"
        "   </IndexedFaceSet>\n"
        "  </Shape>\n"
        " </Scene>\n"
        "</X3D>\n",f);
}

static void write_x3d(const string& filename, RawArray<const Vector<int,3>> tris, RawArray<const TV> X) {
  write_x3d_helper(filename,[=](File& f){
    bool first = true;
    for (const auto& t : tris) {
      if (!first)
        fputc(' ',f);
      first = false;
      fprintf(f,"%d %d %d -1",t.x,t.y,t.z);
    }
  },X);
}
static void write_x3d(const string& filename, const PolygonSoup& soup, RawArray<const TV> X) {
  write_x3d_helper(filename,[&](File& f){
    int offset = 0;
    for (const int c : soup.counts) {
      for (int i=0;i<c;i++)
        fprintf(f,"%d ",soup.vertices[offset++]);
      fputs(offset==soup.vertices.size() ? "-1" : "-1 ",f);
    }
  },X);
}

static Tuple<Ref<TriangleSoup>,Array<TV>> convert(const Tuple<Ref<PolygonSoup>,Array<TV>>& d) {
  return tuple(d.x->triangle_mesh(),d.y);
}

static Tuple<Ref<PolygonSoup>,Array<TV>> convert(const Tuple<Ref<TriangleSoup>,Array<TV>>& d) {
  return tuple(new_<PolygonSoup>(arange(d.x->elements.size()).copy(),scalar_view_own(d.x->elements),d.y.size()),d.y);
}

Tuple<Ref<TriangleSoup>,Array<TV>> read_soup(const string& filename) {
  const auto ext = path::extension(filename);
  if      (ext == ".stl") return         read_stl(filename);
  else if (ext == ".obj") return convert(read_obj(filename));
  else if (ext == ".ply") return convert(read_ply(filename));
  else
    throw ValueError(format("unsupported mesh filename '%s', expected one of .stl, .obj, .ply",filename));
}

Tuple<Ref<PolygonSoup>,Array<TV>> read_polygon_soup(const string& filename) {
  const auto ext = path::extension(filename);
  if      (ext == ".stl") return convert(read_stl(filename));
  else if (ext == ".obj") return         read_obj(filename);
  else if (ext == ".ply") return         read_ply(filename);
  else
    throw ValueError(format("unsupported mesh filename '%s', expected one of .stl, .obj, .ply",filename));
}

Tuple<Ref<TriangleTopology>,Array<TV>> read_mesh(const string& filename) {
  const auto soup = read_soup(filename);
  return tuple(new_<TriangleTopology>(soup.x),soup.y);
}

static void write_helper(const string& filename, RawArray<const Vector<int,3>> tris, RawArray<const TV> X) {
  const auto ext = path::extension(filename);
  if      (ext == ".stl") write_stl(filename,tris,X);
  else if (ext == ".obj") write_obj(filename,tris,X);
  else if (ext == ".ply") write_ply(filename,tris,X);
  else if (ext == ".x3d") write_x3d(filename,tris,X);
  else
    throw ValueError(format("unsupported mesh filename '%s', expected one of .stl, .obj, .ply, .x3d",filename));
}

void write_mesh(const string& filename, const TriangleSoup& soup, RawArray<const TV> X) {
  GEODE_ASSERT(X.size()>=soup.nodes());
  write_helper(filename,soup.elements,X);
}

void write_mesh(const string& filename, const PolygonSoup& soup, RawArray<const TV> X) {
  GEODE_ASSERT(X.size()>=soup.nodes());
  const auto ext = path::extension(filename);
  if      (ext == ".obj") write_obj(filename,soup,X);
  else if (ext == ".ply") write_ply(filename,soup,X);
  else if (ext == ".x3d") write_x3d(filename,soup,X);
  else
    throw ValueError(format("unsupported polygon mesh filename '%s', expected one of .obj, .ply, .x3d",filename));
}

void write_mesh(const string& filename, const TriangleTopology& mesh, RawArray<const TV> X) {
  write_helper(filename,mesh.elements(),X);
}

void write_mesh(const string& filename, const MutableTriangleTopology& mesh) {
  FieldId<Vector<real,3>, VertexId> pos_id(vertex_position_id);
  GEODE_ASSERT(mesh.has_field(pos_id));
  write_helper(filename,mesh.elements(),mesh.field(pos_id).flat);
}

static void write_mesh_py(const string& filename, PyObject* mesh, RawArray<const TV> X) {
  if (auto* soup = python_cast<TriangleSoup*>(mesh))
    write_mesh(filename,*soup,X);
  else if (auto* soup = python_cast<PolygonSoup*>(mesh))
    write_mesh(filename,*soup,X);
  else if (auto* top = python_cast<TriangleTopology*>(mesh))
    write_mesh(filename,*top,X);
  else
    throw TypeError(format("expected TriangleSoup or TriangleTopology, got %s",mesh->ob_type->tp_name));
}

}
using namespace geode;

void wrap_mesh_io() {
  GEODE_FUNCTION(read_soup)
  GEODE_FUNCTION(read_polygon_soup)
  GEODE_FUNCTION(read_mesh)
  GEODE_FUNCTION_2(write_mesh,write_mesh_py)
}
