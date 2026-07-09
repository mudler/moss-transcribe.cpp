#ifndef MT_COMMON_HPP
#define MT_COMMON_HPP

#include <cstdarg>
#include <cstdio>
#include <string>

namespace mt {

// Log severity levels. Defined locally so common.* has no dependency on any
// public C API header (unlike moss-tts.cpp, which pulled this from moss_tts.h).
enum mt_log_level { MT_LOG_ERROR, MT_LOG_WARN, MT_LOG_INFO, MT_LOG_DEBUG };

// Log callback signature: receives a level, a null-terminated message and the
// user pointer registered via set_log_callback().
typedef void (*mt_log_cb)(mt_log_level lvl, const char* msg, void* user);

void set_log_callback(mt_log_cb cb, void* user_data);

void log(mt_log_level lvl, const char* fmt, ...);

#define MT_LOGE(...) ::mt::log(::mt::MT_LOG_ERROR, __VA_ARGS__)
#define MT_LOGW(...) ::mt::log(::mt::MT_LOG_WARN,  __VA_ARGS__)
#define MT_LOGI(...) ::mt::log(::mt::MT_LOG_INFO,  __VA_ARGS__)
#define MT_LOGD(...) ::mt::log(::mt::MT_LOG_DEBUG, __VA_ARGS__)

bool   read_file(const std::string& path, std::string* out);
size_t file_size(const std::string& path);
bool   file_exists(const std::string& path);

}  // namespace mt

#endif  // MT_COMMON_HPP
