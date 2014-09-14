// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_RANDOM_H_
#define BASE_RANDOM_H_
#include <cmath>
#include <stdlib.h>
inline double Random1() {
  return static_cast<double>(random()) / RAND_MAX;
}

inline int Sample1(double a) {
  return Random1() < a ? 1 : 0;
}

inline double Sigmoid(double a) {
  return 1.0 / (1 + exp(-a));
}

inline int SigmoidSample(double a) {
  return Sample1(Sigmoid(a));
}

inline int Random(const VReal &data) {
  VReal tmp(data);
  Cumulate(&tmp);
  double u = Random1() * tmp[tmp.size() - 1];
  for (VReal::size_type i = 0; i < tmp.size(); i++) {
    if (tmp[i] > u) {
      return i;
    }
  }
  return static_cast<int>(tmp.size());
}

inline int Random(int k) {
  double u = (static_cast<double>(random()) / RAND_MAX) * k;
  return std::floor(u);
}

inline void UniformSample(int len, VInt* v) {
  for(int i = 0; i < len; ++i) {
    (*v)[random() % v->size()]++;
  }
}
#endif // BASE_RANDOM_H_
