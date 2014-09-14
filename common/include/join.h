// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_JOIN_H_
#define BASE_JOIN_H_
#include "string_util.h"

#include "base.h"

template <typename Iter>
inline Str JoinStr(Iter beg, Iter end, StrC &del) {
  Str str;
  for (Iter it = beg; it != end; ++it) {
    str.append(*it);
    str.append(del);
  }
  return str;
}

Str Join(const LStr &vec, StrC &del);
Str Join(VStrC &vec, StrC &del);
Str Join(VVStrC &vec, StrC &del1, StrC &del2);
Str Join(VRealC &data, StrC &del);
Str Join(VVRealC &data, StrC &del, StrC &del2);
Str Join(VVVRealC &data, StrC &del1, StrC &del2, StrC &del3);

Str Join(VIntC &vec, StrC &del);
Str Join(const SInt &s, StrC &del);

Str Join(VVIntC &data, StrC &del1, StrC &del2);
Str Join(double* str, int len1);
Str Join(double** str, int len1, int len2);

template <typename T>
inline Str MapToStr(T beg, T end) {
  VStr vec1;
  for (T it = beg; it != end; ++it) {
    VStr vec2;
    vec2.push_back(ToStr(it->first));
    vec2.push_back(ToStr(it->second));
    vec1.push_back(Join(vec2, " "));
  }
  return Join(vec1, "\n");
}

template <typename T>
inline Str JoinValue(T beg, T end) {
  VStr vec;
  for (T it = beg; it != end; ++it) {
    vec.push_back(ToStr(it->second));
  }
  return Join(vec, " ");
}

template <typename T>
inline Str JoinKey(T beg, T end) {
  VStr vec;
  for (T it = beg; it != end; ++it) {
    vec.push_back(ToStr(it->first));
  }
  return Join(vec, " ");
}

inline Str MapToStr(const MIntInt &src) {
  return MapToStr(src.begin(), src.end());
}
#endif // BASE_JOIN_H_
