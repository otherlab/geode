//#####################################################################
// Class LogScope
//#####################################################################
#pragma once

#include <other/core/utility/LogEntry.h>
#include <other/core/utility/tr1.h>
#include <string>
#include <vector>
namespace other {

using std::string;
using std::vector;

class LogScope : public LogEntry {
public:
  unordered_map<string,int> entries;
  vector<LogEntry*> children;
  string scope_identifier;

  LogScope(LogEntry* parent, int depth, const string& scope_identifier, const string& name, int& verbosity_level);
  virtual ~LogScope();

  LogEntry* get_stop_time(FILE* log_file);
  string name_to_identifier(const string& name);
  LogEntry* get_new_scope(FILE* log_file,const string& new_name);
  LogEntry* get_new_item(FILE* log_file,const string& new_name);
  LogEntry* get_pop_scope(FILE* log_file);
  void dump_log(FILE* output);
  void dump_names(FILE* output);
};
}
