// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "include/base.h"

#include <algorithm>
#include <numeric>

#include "include/random.h"

void Init(int len, double value, VReal* des) {
  VReal tmp(len, value);
  des->swap(tmp);
}

void Init(int row, int col, double value, VVReal* des) {
  VReal tmp;
  Init(col, value, &tmp);
  VVReal tmp2(row, tmp);
  des->swap(tmp2);
}

void Init(int len1, int len2, int len3, double value, VVVReal* des) {
  VVReal tmp;
  Init(len2, len3, value, &tmp);
  VVVReal tmp2(len1, tmp);
  des->swap(tmp2);
}

void Cumulate(VReal* des) {
  for (VReal::size_type i = 1; i < des->size(); i++) {
    des->at(i) += des->at(i - 1);
  }
}

int Sum(const VInt &src) {
  return std::accumulate(src.begin(), src.end(), 0);
}

double Sum(const VReal &src) {
  return std::accumulate(src.begin(), src.end(), 0.0);
}

void Sum(const VVReal &src, VReal* des) {
  des->resize(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    des->at(i) = Sum(src[i]);
  }
}

void Append(const VReal &src, VReal* des) {
  for (size_t i = 0; i < src.size(); i++) {
    des->push_back(src[i]);
  }
}

void Append(const VVReal &src, VReal* des) {
  for (size_t i = 0; i < src.size(); i++) {
    Append(src[i], des);
  }
}

void Append(const VVVReal &src, VReal* des) {
  for (size_t i = 0; i < src.size(); i++) {
    Append(src[i], des);
  }
}

void Init(int len, int value, VInt* des) {
  VInt tmp(len, value);
  des->swap(tmp);
}

void Init(int row, int col, int value, VVInt* des) {
  VInt tmp;
  Init(col, value, &tmp);
  VVInt tmp2(row, tmp);
  des->swap(tmp2);
}

void Trans(const VVReal &src, VVReal* des) {
  if (src.empty()) {
    return;
  }
  des->resize(src[0].size());
  for (size_t i = 0; i < des->size(); i++) {
    des->at(i).resize(src.size());
  }
  for (size_t i = 0; i < src.size(); i++) {
    for (size_t j = 0; j < src[0].size(); j++) {
      des->at(j)[i] = src[i][j];
    }
  }
}
