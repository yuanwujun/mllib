// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "include/string_util.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>

using std::stringstream;

void SplitStr(const Str &str, char del, VStr* vec) {
  stringstream stream(str); 
  Str buffer;
  while (getline(stream, buffer, del)) {
    vec->push_back(buffer);
  }
}

void SplitStr(const Str &str, const Str &del, VStr* vec) {
  Str::size_type begin_index = 0;
  while (true) {
    const Str::size_type end_index = str.find(del, begin_index);
    if (end_index == Str::npos) {
      vec->push_back(str.substr(begin_index));
      return;
    }
    vec->push_back(str.substr(begin_index, end_index - begin_index));
    begin_index = end_index + del.size();
  }
}

Str TrimStr(const Str &input) {
  Str str;
  Str space(" \t");
  TrimStr(input, space, &str);
  return str;
}

bool IsWhiteSpace(char c, const Str &white_space) {
  for (Str::size_type i = 0; i < white_space.size(); i++) {
    if (white_space[i] == c) {
      return true; 
    }
  }
  return false;
}

void LTrimStr(const Str &input, const Str &white_space, Str* output) {
  Str::size_type i = 0;
  while(i < input.size() && IsWhiteSpace(input[i], white_space)) {
    i++; 
  }
  output->assign(input.substr(i));
}

void RTrimStr(const Str &input, const Str &white_space, Str* output) {
  Str::size_type i = input.size() - 1;
  while(i >= 0 && IsWhiteSpace(input[i], white_space)) {
    i--; 
  }
  output->assign(input.substr(0, i + 1));
}

void TrimStr(const Str &input, const Str &white_space, Str* output) {
  Str str;
  LTrimStr(input, white_space, &str); 
  RTrimStr(str, white_space, output); 
}

bool StartWith(const Str &str, const Str &search) {
  return str.find(search) == 0;
}

bool EndWith(const Str &str, const Str &search) {
  return str.rfind(search) + search.size() == str.size();
}

Str Lower(const Str &src) {
  Str s(src);
  transform(s.begin(), s.end(), s.begin(), tolower);
  return s;
}

Str Upper(const Str &src) {
  Str s(src);
  transform(s.begin(), s.end(), s.begin(), toupper);
  return s;
}

int StrToInt(const Str &str) {
  int num;
  stringstream stream(str);
  stream >> num;
  return num;
}

double StrToReal(const Str &str) {
  double num;
  stringstream stream(str);
  stream >> num;
  return num;
}
