// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_STRING_UTIL_H_
#define BASE_STRING_UTIL_H_
#include <sstream>

#include "type.h"

void SplitStr(const Str &str, char del, VStr* vec);
void SplitStr(const Str &str, const Str &del, VStr* vec);
void TrimStr(const Str &input, const Str &space, Str* output);
Str TrimStr(const Str &input);
bool IsWhiteSpace(char c, const Str &white_space);
bool StartWith(const Str &str, const Str &search);
bool EndWith(const Str &str, const Str &search);

Str Lower(const Str &src);
Str Upper(const Str &src);

template <typename NumType>
inline Str ToStr(NumType num) {
  std::stringstream stream;
  stream << num;
  Str str;
  stream >> str;
  return str;
}

template <typename NumType>
inline Str ToStr(NumType num, int precision) {
  std::stringstream stream;
  stream.precision(precision);
  stream << num;
  Str str;
  stream >> str;
  return str;
}

int StrToInt(const Str &str);
double StrToReal(const Str &str);
#endif  // BASE_STRING_UTIL_H_
