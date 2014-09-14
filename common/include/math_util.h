// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_MATH_UTIL_H_
#define BASE_MATH_UTIL_H_
#include <cmath>
inline void Subtract(double m, VReal* v) {
  for (size_t i = 0; i < v->size(); i++) {
    v->at(i) -= m;
  }
}

inline void Exp(VReal* v) {
  for (size_t i = 0; i < v->size(); i++) {
    v->at(i) = exp(v->at(i));
  }
}

inline double Factorial(int n) {
  double r = 1;
  for (int i = 1; i <= n; i++) {
    r *= i;
  }
  return r;
}

inline double MultiNum(int len, const VInt &v) {
  double a = Factorial(len);
  for (size_t i = 0; i < v.size(); i++) {
    a /= Factorial(v[i]);
  }
  return a;
}

inline double Log2(double a) {
  return log(a) / log(2.0);
}
#endif  // BASE_MATH_UTIL_H_
