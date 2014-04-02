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
#include <geode/utility/Unique.h>
#include <sstream>
#include <iostream>
namespace geode {

using std::string;
using std::stringbuf;
using std::streambuf;
using std::ostream;
static void dump_log_helper();

namespace {
FILE* log_file=0;
Unique<streambuf> cout_buffer;
Unique<streambuf> cerr_buffer;
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
  Unique<T> object;
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
Unique<LogClass> private_instance;

static void initialize() {
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

class LogCerrBuffer : public stringbuf {
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
  Log::cout.rdbuf(cout_buffer.get());
  cerr_buffer.reset(new LogCerrBuffer);
  Log::cerr.rdbuf(cerr_buffer.get());
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
    Log::cout.rdbuf(std::cout.rdbuf());
  if (cerr_initialized)
    Log::cerr.rdbuf(std::cerr.rdbuf());
  delete root;
  root = 0;
}

} // unnamed namespace

void log_cache_initial_output() {
  if (!log_file) {
    log_file = tmpfile();
    if (!log_file) GEODE_FATAL_ERROR("Couldn't create temporary log file");
    log_file_temporary = true;
  }
}

namespace Log {
ostream& cout_helper() {
  static InitializationHelper<ostream> helper(new ostream(std::cout.rdbuf()),cout_initialized); // Necessary for DLLs to work. Cannot use static class data across dlls
  return *helper.object;
}

ostream& cerr_helper() {
  static InitializationHelper<ostream> helper(new ostream(std::cerr.rdbuf()),cerr_initialized); // Necessary for DLLs to work. Cannot use static class data across dlls
  return *helper.object;
}
}

bool log_initialized() {
  return bool(private_instance);
}

void log_configure(const string& name, const bool suppress_cout_in, const bool suppress_timing_in, const int verbosity) {
  root_name = name;
  suppress_cout = suppress_cout_in;
  suppress_cerr = false;
  suppress_timing = suppress_timing_in;
  verbosity_level = verbosity-1;
  initialize();
}

void log_copy_to_file(const string& filename, const bool append) {
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

void log_finish() {
  private_instance.reset();
}

bool log_is_timing_suppressed() {
  initialize();
  return suppress_timing;
}

void log_time_helper(const string& label) {
  // Always called after is_timing_suppressed, so no need to call initialized()
  current_entry = current_entry->get_new_item(log_file,label);
  current_entry->start(log_file);
}

void log_stop_time() {
  if (!log_is_timing_suppressed())
    current_entry=current_entry->get_stop_time(log_file);
}

template<class TValue> void log_stat(const string& label, const TValue& value) {
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

template void log_stat(const string&,const int&);
template void log_stat(const string&,const bool&);
template void log_stat(const string&,const float&);
template void log_stat(const string&,const double&);

void log_push_scope(const string& name) {
  initialize();
  if (suppress_timing) return;
  current_entry = current_entry->get_new_scope(log_file,name);
  current_entry->start(log_file);
}

void log_pop_scope() {
  if (!current_entry) return;
  initialize();
  if (suppress_timing) return;
  current_entry = current_entry->get_pop_scope(log_file);
}

void log_reset() {
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

void log_dump() {
  initialize();
  dump_log_helper();
}

void log_print(const string& str) {
  Log::cout << str << std::endl;
}

void log_error(const string& str) {
  Log::cerr << str << std::endl;
}

void log_flush() {
  Log::cout << std::flush;
}

}
