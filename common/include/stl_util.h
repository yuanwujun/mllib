// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_STL_UTIL_H_
#define BASE_STL_UTIL_H_
#include "stl_util.h"

inline void Add(const VInt &src, VInt* des) {
  for (size_t i = 0; i < src.size(); i++) {
    des->at(i) += src[i];
  }
}

inline void ToSet(const VInt &src, SInt* des) {
  for (size_t i = 0; i < src.size(); i++) {
    des->insert(src[i]);
  }
}

inline int DiffNum(const VInt &lhs, const VInt &rhs) {
  int count = 0;
  for (size_t i = 0; i < lhs.size(); i++) {
    if (lhs[i] != rhs[i]) {
      count++;
    }
  }
  return count;
}

inline void Multiply(const VReal &src, double m, VReal* des) {
  for (size_t i = 0; i < src.size(); i++) {
    des->at(i) = src.at(i) * m;
  }
}

inline void Multiply(const VVReal &src, double m, VVReal* des) {
  for (size_t i = 0; i < src.size(); i++) {
    Multiply(src[i], m, &(des->at(i)));
  }
}

template <typename T>
inline double Max(const T &data) {
  return *(std::max_element(data.begin(), data.end()));
}

template <typename E, typename C>
inline void Push(int num, const E &e, C* des) {
  for (int i = 0; i < num; i++) {
    des->push_back(e);
  }
}
#endif // BASE_STL_UTIL
