// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "include/join.h"
#include "include/string_util.h"

Str Join(const VStr &vec, const Str &del) {
  return JoinStr(vec.begin(), vec.end(), del);		   
}

Str Join(const LStr &l, const Str &del) {
  return JoinStr(l.begin(), l.end(), del);		   
}

Str Join(const VVStr &vec, const Str &del1, const Str &del2) {
  VStr tmp;
  for (VVStr::size_type i = 0; i < vec.size(); i++) {
    tmp.push_back(Join(vec.at(i), del1));
  }
  return Join(tmp, del2);
}

template <typename It>
Str Join(It beg, It end, StrC &del) {
  VStr tmp;
  for (It it = beg; it != end; ++it) {
    tmp.push_back(ToStr(*it, 7));
  }
  return Join(tmp, del);
}

Str Join(const VInt &data, const Str &del) {
  return Join(data.begin(), data.end(), del);
}

Str Join(const SInt &data, StrC &del) {
  return Join(data.begin(), data.end(), del);
}

Str Join(VVIntC &data, StrC &del1, StrC &del2) {
  VStr tmp;
  for (VVInt::size_type i = 0; i < data.size(); i++) {
    tmp.push_back(Join(data.at(i), del1));
  }
  return Join(tmp, del2);
}

Str Join(const VReal &data, const Str &del) {
  return Join(data.begin(), data.end(), del);
}

Str Join(const VVReal &data, const Str &del1, const Str &del2) {
  VStr tmp;
  for (VVReal::size_type i = 0; i < data.size(); i++) {
    tmp.push_back(Join(data.at(i), del1));
  }
  return Join(tmp, del2);
}

Str Join(double* str, int len) {
  VStr tmp;
  for (int i = 0; i < len; i++) {
    tmp.push_back(ToStr(str[i]));
  }
  return Join(tmp, " ");
}

Str Join(double** str, int len1, int len2) {
  VStr tmp;
  for (int i = 0; i < len1; i++) {
    tmp.push_back(Join(str[i], len2));
  }
  return Join(tmp, "\n");
}

Str Join(const VVVReal &data, StrC &del1, StrC &del2, StrC &del3) {
  VStr tmp;
  for (VVVReal::size_type i = 0; i < data.size(); i++) {
    tmp.push_back(Join(data.at(i), del1, del2));
  }
  return Join(tmp, "\n");
}
