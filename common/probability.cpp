// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "include/probability.h"
#include <algorithm>
#include <numeric>

int SumTopN(const VInt &src, int len) {
  return std::accumulate(src.begin(), src.begin() + len, 0);
}

bool NextMultiSeq(VInt* des) {
  if (SumTopN(*des, des->size() - 1) == 0) {
    return false;
  }
  int pos = 0;
  while (des->at(pos) == 0) {
    pos++;
  }
  if (pos == 0) {
    des->at(0) -= 1;
    des->at(1) += 1;
    return true;
  } else {
    des->at(0) = des->at(pos) - 1;
    des->at(pos) = 0;
    des->at(pos + 1) += 1;
  }
  return true;
}

bool NextBinarySeq(VInt* des) {
  if (des->at(0) == 0) {
    des->at(0) = 1;
    return true;
  }
  size_t i = 0;
  while (i < des->size() && des->at(i) == 1) {
    des->at(i) = 0;
    i++;
  }
  if (i < des->size()) {
    des->at(i) = 1;
    return true;
  } else {
    return false;
  }
}
