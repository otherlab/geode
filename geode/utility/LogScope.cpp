//#####################################################################
// Class LogScope
//#####################################################################
#include <geode/utility/LogScope.h>
namespace geode {

LogScope::LogScope(LogEntry* parent, int depth, const string& scope_identifier, const string& name, int& verbosity_level)
  : LogEntry(parent,depth,name,verbosity_level), scope_identifier(scope_identifier) {}

LogScope::~LogScope() {
  for (auto child : children)
    delete child;
}

LogEntry* LogScope::get_stop_time(FILE* log_file) {
  return this;
}

string LogScope::name_to_identifier(const string& name) {
  // Extract portion of name up to first digit
  size_t size = name.size();
  for (size_t i=0;i<size;i++)
    if (isdigit(name[i])){
      size=i;
      break;}
  // Remove trailing whitespace
  while (size && isspace(name[size-1]))
    size--;
  return name.substr(0,size);
}

LogEntry* LogScope::get_new_scope(FILE* log_file,const string& new_name) {
  end_on_separate_line = true;
  log_file_end_on_separate_line = true;
  string new_scope_identifier = name_to_identifier(new_name);
  const auto entry = entries.find(new_scope_identifier);
  if (entry!=entries.end()) {
    children[entry->second]->name = new_name;
    return children[entry->second];
  }
  LogEntry* new_entry = new LogScope(this,depth+1,new_scope_identifier,new_name,verbosity_level);
  children.push_back(new_entry);
  entries[new_scope_identifier] = (int)children.size()-1;
  return new_entry;
}

LogEntry* LogScope::get_new_item(FILE* log_file,const string& new_name) {
  end_on_separate_line = true;
  log_file_end_on_separate_line = true;
  const auto entry = entries.find(new_name);
  if (entry!=entries.end())
    return children[entry->second];
  LogEntry* new_entry = new LogEntry(this,depth+1,new_name,verbosity_level);
  children.push_back(new_entry);
  entries[new_name] = (int)children.size()-1;
  return new_entry;
}

LogEntry* LogScope::get_pop_scope(FILE* log_file) {
  stop(log_file);
  return parent;
}

void LogScope::dump_log(FILE* output) {
  fprintf(output,"%*s%-*s%8.4f s\n",2*depth,"",50-2*depth,scope_identifier.c_str(),time);
  fflush(output);
  for (auto child : children)
    child->dump_log(output);
}

void LogScope::dump_names(FILE* output) {
  LogEntry::dump_names(output);
  for (auto child : children)
    child->dump_names(output);
}

}
