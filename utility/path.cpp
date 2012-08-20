// Path convenience functions

#include <other/core/utility/path.h>
namespace other {
namespace path {

string extension(const string& path) {
  for (int i=path.size()-1;i>=0;i--) {
#ifdef _WIN32
    if (path[i]=='/' || path[i]=='\\')
#else
    if (path[i]=='/')
#endif
      break;
    else if (path[i]=='.')
      return path.substr(i);
  }
  return string();
}

}
}
