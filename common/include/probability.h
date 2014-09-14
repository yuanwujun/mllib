// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_PROBABILITY_H_
#define BASE_PROBABILITY_H_
#include "type.h"
#include "math_util.h"
int SumTopN(const VInt &src, int len);
bool NextMultiSeq(VInt* des);
bool NextBinarySeq(VInt* des);

inline double LogBer(int t, double p) {
  return t*Log2(p) + (1 - t)*Log2(1 - p);
}
#endif // BASE_PROBABILITY_H_
