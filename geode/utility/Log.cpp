//#####################################################################
// Class Log
//#####################################################################
#include <geode/utility/Log.h>
#include <geode/utility/debug.h>
#include <geode/utility/format.h>
#include <geode/utility/LogEntry.h>
#include <geode/utility/LogScope.h>
#include <geode/utility/str.h>
#include <geode/utility/time.h>
#include <boost/scoped_ptr.hpp>
#include <sstream>
#include <iostream>
namespace geode {
namespace Log {

using std::string;
using std::stringbuf;
using std::streambuf;
using std::ostream;
using boost::scoped_ptr;

namespace {
FILE* log_file=0;
scoped_ptr<streambuf> cout_buffer;
scoped_ptr<streambuf> cerr_buffer;
bool cout_initialized = false;
bool cerr_initialized = false;

string root_name;
bool suppress_cout = false;
bool suppress_cerr = false;
int verbosity_level = 1<<30;
bool log_file_temporary = false;
LogEntry* root = 0;
LogEntry* current_entry = 0;

bool suppress_timing;

template<class T>
struct InitializationHelper {
  scoped_ptr<T> object;
  bool& initialized;

  InitializationHelper(T* object, bool& initialized)
    : object(object), initialized(initialized) {
    initialized = true;
  }

  ~InitializationHelper() {
    initialized=false;
  }
};

struct LogClass {
  LogClass();
  ~LogClass();
};
scoped_ptr<LogClass> private_instance;

}
static void dump_log_helper();

void initialize() {
  if(!private_instance) new LogClass();
}

class LogCoutBuffer : public stringbuf {
  int sync() {
    initialize();
    if (!suppress_cout && current_entry->depth<verbosity_level) {
      if (LogEntry::start_on_separate_line)
        putchar('\n');
      string buffer = str();
      for (size_t start=0;start<buffer.length();) {
        size_t end = buffer.find('\n',start);
        if (LogEntry::needs_indent) {
          printf("%*s",2*current_entry->depth+2,"");
          LogEntry::needs_indent = false;
        }
        fputs(buffer.substr(start,end-start).c_str(),stdout);
        if (end!=string::npos) {
          putchar('\n');
          LogEntry::needs_indent = true;
          start=end+1;
        } else
          break;
      }
      LogEntry::start_on_separate_line = false;
      current_entry->end_on_separate_line = true;
      fflush(stdout);
    }
    if (log_file) {
      if (LogEntry::log_file_start_on_separate_line)
        putc('\n',log_file);
      string buffer=str();
      for (size_t start=0;start<buffer.length();) {
        size_t end = buffer.find('\n',start);
        if (LogEntry::log_file_needs_indent) {
          fprintf(log_file,"%*s",2*current_entry->depth+2,"");
          LogEntry::log_file_needs_indent = false;
        }
        fputs(buffer.substr(start,end-start).c_str(),log_file);
        if (end!=string::npos) {
          putc('\n',log_file);
          LogEntry::log_file_needs_indent = true;
          start = end+1;
        } else
          break;
      }
      LogEntry::log_file_start_on_separate_line = false;
      current_entry->log_file_end_on_separate_line = true;
      fflush(log_file);
    }
    str("");
    return stringbuf::sync();
  }
};

class LogCerrBuffer:public stringbuf {
  int sync() {
    initialize();
    if (!suppress_cerr) {
      if (LogEntry::start_on_separate_line)
        putchar('\n');
      LogEntry::start_on_separate_line = false;
      fputs(str().c_str(),stderr);
    }
    if (log_file) {
      if (LogEntry::log_file_start_on_separate_line)
        putc('\n',log_file);
      LogEntry::log_file_start_on_separate_line = false;
      string buffer = str();
      for (size_t start=0;start<buffer.length();) {
        size_t end = buffer.find('\n',start);
        fputs(buffer.substr(start,end-start).c_str(),log_file);
        putc('\n',log_file);
        if (end!=string::npos)
          start=end+1;
        else
          break;
      }
    }
    str("");
    return stringbuf::sync();
  }
};

LogClass::LogClass() {
  private_instance.reset(this);
  cout_buffer.reset(new LogCoutBuffer);
  cout.rdbuf(cout_buffer.get());
  cerr_buffer.reset(new LogCerrBuffer);
  cerr.rdbuf(cerr_buffer.get());
  root = new LogScope(0,0,root_name,root_name,verbosity_level);
  current_entry = root;
  root->start(log_file);
}

LogClass::~LogClass() {
  while (current_entry!=0)
    current_entry=current_entry->get_pop_scope(log_file);
  dump_log_helper();
  if (log_file)
    fclose(log_file);
  log_file = 0;
  log_file_temporary = false;
  if (cout_initialized)
    cout.rdbuf(std::cout.rdbuf());
  if (cerr_initialized)
    cerr.rdbuf(std::cerr.rdbuf());
  delete root;
  root = 0;
}

void cache_initial_output() {
  if (!log_file) {
    log_file = tmpfile();
    if (!log_file) GEODE_FATAL_ERROR("Couldn't create temporary log file");
    log_file_temporary = true;
  }
}

ostream& cout_Helper() {
  static InitializationHelper<ostream> helper(new ostream(std::cout.rdbuf()),cout_initialized); // Necessary for DLLs to work. Cannot use static class data across dlls
  return *helper.object;
}

ostream& cerr_Helper() {
  static InitializationHelper<ostream> helper(new ostream(std::cerr.rdbuf()),cerr_initialized); // Necessary for DLLs to work. Cannot use static class data across dlls
  return *helper.object;
}

bool initialized() {
  return bool(private_instance);
}

void configure(const string& root_name_input, const bool suppress_cout_input, const bool suppress_timing_input, const int verbosity_level_input) {
  root_name = root_name_input;
  suppress_cout = suppress_cout_input;
  suppress_cerr = false;
  suppress_timing = suppress_timing_input;
  verbosity_level = verbosity_level_input-1;
  initialize();
}

void copy_to_file(const string& filename,const bool append) {
  initialize();
  FILE* temporary_file = 0;
  if (log_file && log_file_temporary){
    temporary_file = log_file;
    log_file=0;
  }
  if (log_file) {
    if (LogEntry::log_file_start_on_separate_line)
      putc('\n',log_file);
    root->dump_log(log_file);
    fclose(log_file);
    log_file=0;
  }
  if (!filename.empty()) {
    if (append) {
      log_file = fopen(filename.c_str(),"a");
      if (!log_file) GEODE_FATAL_ERROR(format("Can't open log file %s for append",filename));
      putc('\n',log_file);
    } else {
      log_file = fopen(filename.c_str(),"w");
      if(!log_file) GEODE_FATAL_ERROR(format("Can't open log file %s for writing",filename));
    }
    if (temporary_file) {
      fflush(temporary_file);
      fseek(temporary_file,0,SEEK_SET);
      string buffer;
      buffer.resize(4096);
      for (;;) {
        size_t n = fread(&buffer[0],sizeof(char),buffer.size(),temporary_file);
        fwrite(buffer.c_str(),sizeof(char),n,log_file);
        if(n<buffer.size()) break;
      }
      fflush(log_file);
    } else if (private_instance) {
      root->dump_names(log_file);
      LogEntry::log_file_start_on_separate_line = LogEntry::log_file_needs_indent = current_entry->log_file_end_on_separate_line=true;
    }
  }
  if(temporary_file) fclose(temporary_file);
  log_file_temporary = false;
}

void finish() {
  private_instance.reset();
}

bool is_timing_suppressed() {
  initialize();
  return suppress_timing;
}

void time_helper(const string& label) {
  // Always called after is_timing_suppressed, so no need to call initialized()
  current_entry = current_entry->get_new_item(log_file,label);
  current_entry->start(log_file);
}

void stop_time() {
  if(!is_timing_suppressed())
    current_entry=current_entry->get_stop_time(log_file);
}

template<class TValue> void stat(const string& label, const TValue& value) {
  initialize();
  if (suppress_timing) return;
  string s = str(value);
  if (current_entry->depth<verbosity_level) {
    if (LogEntry::start_on_separate_line) putchar('\n');
    if (LogEntry::needs_indent) printf("%*s",2*current_entry->depth+2,"");
    printf("%s = %s\n",label.c_str(),s.c_str());
    LogEntry::start_on_separate_line = false;
    LogEntry::needs_indent = current_entry->end_on_separate_line = true;
  }
  if (log_file) {
    if (LogEntry::log_file_start_on_separate_line) putc('\n',log_file);
    if (LogEntry::log_file_needs_indent) fprintf(log_file,"%*s",2*current_entry->depth+2,"");
    fprintf(log_file,"%s = %s\n",label.c_str(),s.c_str());
    LogEntry::log_file_start_on_separate_line = false;
    LogEntry::log_file_needs_indent = current_entry->log_file_end_on_separate_line = true;
  }
}

template void stat(const string&,const int&);
template void stat(const string&,const bool&);
template void stat(const string&,const float&);
template void stat(const string&,const double&);

void push_scope(const string& name) {
  initialize();
  if (suppress_timing) return;
  current_entry = current_entry->get_new_scope(log_file,name);
  current_entry->start(log_file);
}

void pop_scope() {
  if (!current_entry) return;
  initialize();
  if (suppress_timing) return;
  current_entry = current_entry->get_pop_scope(log_file);
}

void reset() {
  initialize();
  if (suppress_timing) return;
  delete root;
  root = new LogScope(0,0,"simulation","simulation",verbosity_level);
  current_entry=root;
}

static void dump_log_helper() {
  if (suppress_timing) return;
  if (LogEntry::start_on_separate_line) {
    putchar('\n');
    LogEntry::start_on_separate_line = false;
  }
  if (!suppress_cout) root->dump_log(stdout);
  if (log_file) {
    if (LogEntry::log_file_start_on_separate_line) {
      putc('\n',log_file);
      LogEntry::log_file_start_on_separate_line = false;
    }
    root->dump_log(log_file);
  }
}

void dump_log() {
  initialize();
  dump_log_helper();
}

}
}
