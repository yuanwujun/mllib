// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_STAT_H_
#define BASE_STAT_H_
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cmath>

#include "stl_util.h"
#include "math_util.h"

template <typename V, typename M>
inline double Quadratic(const V &x, const V &y, const M &w) {
  double sum = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    for (size_t j = 0; j < y.size(); ++j) {
      sum += x[i] * w[i][j] * y[j];
    }
  }
  return sum;
}

template <typename V1, typename V2>
inline double InnerProd(const V1 &x, const V2 &y) {
  double sum = 0;
  for (size_t i = 0; i < x.size(); i++) {
    sum += x[i] * y[i];
  }
  return sum;
}

inline double LogSum(double log_a, double log_b) {
  double v;
  if (log_a < log_b) {
    v = log_b + log(1 + exp(log_a - log_b));
  } else {
    v = log_a + log(1 + exp(log_b - log_a));
  }
  return v;
}

//input: the energy vector
//output: log partition
inline double LogPartition(const VReal &data) {
  double m = Max(data);
  VReal tmp(data);
  Subtract(m, &tmp);
  Exp(&tmp);
  return m + log(Sum(tmp));
}

inline double LogPartition(const VInt &num, const VReal &data) {
  double m = Max(data);
  VReal tmp(data);
  Subtract(m, &tmp);
  Exp(&tmp);
  return m + log(InnerProd(num, tmp));
}

/**
   * Proc to calculate the value of the trigamma, the second
   * derivative of the loggamma function. Accepts positive matrices.
   * From Abromowitz and Stegun.  Uses formulas 6.4.11 and 6.4.12 with
   * recurrence formula 6.4.6.  Each requires workspace at least 5
   * times the size of X.
   **/
inline double TriGamma(double x) {
  x = x + 6;
  double p = 1/(x*x);
  p= (((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
         *p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
  for (int i = 0; i < 6; i++) {
    x = x - 1;
    p = 1/(x*x)+p;
  }
  return p;
}

/* taylor approximation of first derivative of the log gamma function*/
inline double DiGamma(double x) {
  double p;
  x = x + 6;
  p = 1/(x*x);
  p = (((0.004166666666667*p-0.003968253986254)*p+
         0.008333333333333)*p-0.083333333333333)*p;
  p = p + log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
  return p;
}

inline double LogGamma(double x) {
  double z = 1/(x*x);
  x = x + 6;
  z = (((-0.000595238095238*z+0.000793650793651)
      *z-0.002777777777778)*z+0.083333333333333)/x;
  z = (x-0.5)*log(x)-x+0.918938533204673+z-log(x-1)-
      log(x-2)-log(x-3)-log(x-4)-log(x-5)-log(x-6);
  return z;
}

inline void Probability(const VReal &lhs, VReal* des) {
  double sum = std::accumulate(lhs.begin(), lhs.end(), 0.0);
  des->resize(lhs.size());
  for(VReal::size_type i = 0; i < lhs.size(); i++) {
    des->at(i) = lhs.at(i) / sum;
  }
}

inline double Expect(const VReal &pro) {
  double e = 0;
  for(VReal::size_type i = 0; i < pro.size(); i++) {
    e += pro.at(i) * (i + 1);
  }
  return e;
}
#endif  // BASE_STAT_H_
