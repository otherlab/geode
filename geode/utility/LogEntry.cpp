#include <geode/utility/LogEntry.h>
namespace geode {

bool LogEntry::start_on_separate_line=false;
bool LogEntry::log_file_start_on_separate_line=false;
bool LogEntry::needs_indent=true;
bool LogEntry::log_file_needs_indent=true;

LogEntry::LogEntry(LogEntry* parent, const int depth, const string& name, int& verbosity_level)
  : parent(parent), depth(depth), time(0), name(name), verbosity_level(verbosity_level) {
  end_on_separate_line = false;
  log_file_end_on_separate_line = false;
  timer_start_time = get_time();
}

LogEntry::~LogEntry() {}

void LogEntry::start(FILE* log_file) {
  if (depth<=verbosity_level) {
    if (start_on_separate_line) putchar('\n');
    start_on_separate_line = needs_indent = true;
    printf("%*s%-*s",2*depth,"",50-2*depth,name.c_str());
    fflush(stdout);
  }
  if (log_file) {
    if (log_file_start_on_separate_line) putc('\n',log_file);
    log_file_start_on_separate_line = log_file_needs_indent = true;
    fprintf(log_file,"%*s%-*s",2*depth,"",50-2*depth,name.c_str());
    fflush(log_file);
  }
  timer_start_time = get_time();
}

void LogEntry::stop(FILE* log_file) {
  double time_since_start = get_time()-timer_start_time;
  if (depth<=verbosity_level) {
    if (end_on_separate_line) {
      if(start_on_separate_line) putchar('\n');
      printf("%*sEND %-*s",2*depth,"",50-2*depth-4,name.c_str());
    }
    end_on_separate_line = start_on_separate_line = false;
    needs_indent = true;
    printf("%8.4f s\n",time_since_start);
    fflush(stdout);
  }
  if (log_file) {
    if (log_file_end_on_separate_line) {
      if (log_file_start_on_separate_line) putc('\n',log_file);
      fprintf(log_file,"%*sEND %-*s",2*depth,"",50-2*depth-4,name.c_str());
    }
    log_file_end_on_separate_line = log_file_start_on_separate_line = false;
    log_file_needs_indent = true;
    fprintf(log_file,"%8.4f s\n",time_since_start);
    fflush(log_file);
  }
  time += time_since_start;
}

LogEntry* LogEntry::get_stop_time(FILE* log_file) {
  stop(log_file);
  return parent;
}

LogEntry* LogEntry::get_new_scope(FILE* log_file,const string& new_name) {
  stop(log_file);
  return parent->get_new_scope(log_file,new_name);
}

LogEntry* LogEntry::get_new_item(FILE* log_file,const string& new_name) {
  stop(log_file);
  return parent->get_new_item(log_file,new_name);
}

LogEntry* LogEntry::get_pop_scope(FILE* log_file) {
  stop(log_file);
  return parent->get_pop_scope(log_file);
}

void LogEntry::dump_log(FILE* output) {
  fprintf(output,"%*s%-*s%8.4f s\n",2*depth,"",50-2*depth,name.c_str(),time);
  fflush(output);
}

void LogEntry::dump_names(FILE* output) {
  fprintf(output,"%*s%-*s",2*depth,"",50-2*depth,name.c_str());
  fflush(output);
}

}
