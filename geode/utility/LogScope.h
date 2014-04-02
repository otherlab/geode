//#####################################################################
// Class LogScope
//#####################################################################
#pragma once

#include <geode/utility/LogEntry.h>
#include <geode/utility/unordered.h>
#include <string>
#include <vector>
namespace geode {

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
