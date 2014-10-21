// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_IO_UTIL_H_
#define BASE_IO_UTIL_H_
#include <iostream> 
#include <fstream>
//#include <stdio.h>
#include <stdlib.h>

#include "string_util.h"

inline void ReadFileToStr(const Str &file, Str* str) {
  std::ifstream in(file.c_str());
  std::istreambuf_iterator<char> beg(in);
  std::istreambuf_iterator<char> end;
  str->assign(beg, end);
  in.close();
}

inline void ReadFileToStr(const Str &file, const Str &del, VStr* data) {
  Str str;
  ReadFileToStr(file, &str);
  SplitStr(str, del, data);
}

inline Str ReadFileToStr(const Str &file) {
  Str str;
  ReadFileToStr(file, &str);
  return str;
}

inline void WriteStrToFile(const Str &str, const Str &file) {
  std::ofstream o(file.c_str());
  o << str;
  o.close();
}

inline void AppendStrToFile(const Str &str, const Str &file) {
  std::ofstream o(file.c_str(),std::ofstream::out|std::ofstream::app);
  o << str << std::endl;
  o.close();
}

inline void ReadFile(const Str &file, VInt* des) {
  std::ifstream in(file.c_str(), std::ios::binary);
  in.read((char*)(&(des->at(0))), sizeof(*des));
  in.close( );
}

inline void WriteFile(const Str &file, const VInt &data) {
  std::ofstream out(file.c_str(), std::ios::binary);
  out.write((char*)(&data[0]), sizeof(data));
  out.close();
}

inline bool IsFile(const Str &path) {
  std::fstream file;
  file.open(path.c_str(), std::ios::in);
  if(!file) {
    file.close();
    return false;
  } else {
    file.close();
    return true;
  }
}
#endif  // BASE_IO_UTIL_H_
