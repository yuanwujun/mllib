// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_BASE_H_
#define BASE_BASE_H_
#include <cmath>
#include "type.h"
void Init(int len, double value, VReal* des);
void Init(int row, int col, double value, VVReal* des);
void Init(int len1, int len2, int len3, double value, VVVReal* des);

void Init(int len, int value, VInt* des);
void Init(int row, int col, int value, VVInt* des);
void Cumulate(VReal* des);

inline double Square(double a) {
  return a * a;
}

void Append(const VReal &src, VReal* des);
void Append(const VVReal &src, VReal* des);
void Append(const VVVReal &src, VReal* des);

int Sum(const VInt &src);
double Sum(const VReal &src);
void Sum(const VVReal &src, VReal* des);

template<typename Iterator,typename Num_Type>
inline void Range(Iterator t, Num_Type beg, int length, Num_Type interval = 1) {
  for(int i = 0; i < length; i++) {
    *(t++) = beg;
    beg += interval;
  }
}

inline void Range(int beg, int end, int interval, VInt* s) {
  for (int i = beg; i != end; i += interval) {
    s->push_back(i); 
  }
}

inline void Range(double beg, double end, double interval, VReal* s) {
  for (double i = beg; i < end; i += interval) {
    s->push_back(i); 
  }
}

class Time {
 public:
  void Start() { beg = clock(); }
  double GetTime() {
    return static_cast<double>(clock() - beg) / CLOCKS_PER_SEC;
  }
 private:
  int beg;
};

void Trans(const VVReal &src, VVReal* des);
 
#endif  // BASE_BASE_H_
