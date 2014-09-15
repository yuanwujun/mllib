// Copyright 2014 lijiankou. All Rights Reserved.
// author: lijk_start@163.com (jiankou li)
#ifndef ML_EIGEN_UTIL_H_
#define ML_EIGEN_UTIL_H_
#include "eigen.h"

//input: the energy vector
//output: log sum 
inline double LogSum(const Vec &data) {
  double m = data.maxCoeff();
  Vec t(data.size());
  t.setConstant(m);
  Vec tmp = data - t;
  for (int i = 0; i < tmp.size(); i++) {
    tmp[i] = exp(tmp[i]);
  }
  return m + log(tmp.sum());
}

#endif // ML_EIGEN_UTIL_H_
