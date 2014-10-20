// copyright 2014 lijiankou. all rights reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef TOPIC_UTIL_H_
#define TOPIC_UTIL_H_
#include "base_head.h"
#include "cokus.h"
#include "eigen.h"

namespace ml {
inline void GThetaEp(VecC &vec, Vec* des) {
  double digsum = DiGamma(vec.sum());
  des->resize(vec.size());
  for (int k = 0; k < vec.size(); k++) {
    (*des)[k] = DiGamma(vec[k]) - digsum;
  }
}

inline void LThetaEp(MatC &mat, Mat* des) {
  des->resize(mat.rows(), mat.cols());
  for (int j = 0; j < mat.cols(); j++) {
    double digsum = DiGamma(mat.col(j).sum());
    for (int k = 0; k < mat.rows(); k++) {
      (*des)(k, j) = DiGamma(mat(k, j)) - digsum;
    }
  }
}

inline void OmegaEp(VecC &vec, Vec* des) {
  des->resize(vec.size());
  double sum = DiGamma(vec.sum());
  (*des)[0] = DiGamma(vec[0]) - sum;
  (*des)[1] = DiGamma(vec[1]) - sum;
}

inline double LogDelta(VecC &alpha) {
  double res = 0;
  for (int i = 0; i < alpha.size(); i++) {
    res += lgamma(alpha[i]);
  }
  return res - lgamma(alpha.sum()); 
}

inline double LogBeta(VecC &vec) {
  return lgamma(vec[0]) + lgamma(vec[1]) - lgamma(vec.sum());
}

inline double LogDelta(int num, double alpha) {
  return num * lgamma(alpha) - lgamma(num * alpha);
}
} // namespace ml
#endif // TOPIC_UTIL_H_
