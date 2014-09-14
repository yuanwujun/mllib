// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_C_UTIL_H_
#define BASE_C_UTIL_H_
#include <algorithm>
inline int Max(double* x, int n) {
  return static_cast<int>(*std::max_element(x, x + n));
}

inline double** NewArray(int row, int col) {
  double** des = NULL;
  des = new double*[row];
  for (int i = 0; i < row; i++) {
    des[i] = new double[col];
  }
  return des;
}

inline void DelArray(double** a, int row) {
  for (int i = 0; i < row; i++) {
    delete [] a[i];
  }
  delete [] a;
}

inline void Init(int len, double value, double* des) {
  for (int i = 0; i < len; i++) {
    des[i] = value;
  }
}

inline void Init(int row, int col, double value, double** des) {
  for (int i = 0; i < row; i++) {
    Init(col, value, des[i]);
  }
}
#endif  // BASE_C_UTIL_H_
