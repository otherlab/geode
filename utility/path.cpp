// Path convenience functions

#include <other/core/utility/path.h>
namespace other {
namespace path {

string join(const string& p, const string& q) {
  return p+sep+q;
}

string extension(const string& path) {
  for (int i=path.size()-1;i>=0;i--) {
    if (is_sep(path[i]))
      break;
    if (path[i]=='.')
      return path.substr(i);
  }
  return string();
}

string remove_extension(const string& path) {
  for (int i=path.size()-1;i>=0;i--) {
    if (is_sep(path[i])) 
      break;
    if (path[i]=='.')
      return path.substr(0,i); 
  }
  return path;
}

string basename(const string& path) {
  for (int i=path.size()-1;i>=0;i--)
    if (is_sep(path[i]))
      return path.substr(i+1);
  return path;
}

}
}
