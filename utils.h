#ifndef __UTILS_H_
#define __UTILS_H_

#include <string>

char* get_cmd_option(char ** begin, char ** end, const std::string & option);

bool cmd_option_exists(char** begin, char** end, const std::string& option);

#endif