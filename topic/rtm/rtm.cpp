// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "rtm.h"

namespace ml {
void RTMSuffStats::SetZero(int k, int v) { 
  topic.resize(k, v);
  topic.setZero();
  topic_sum.resize(k);
  topic_sum.setZero();
}

void RTMSuffStats::InitSS(int k, int v) { 
  topic.resize(k, v);
  topic.setConstant(1.0 / v);
  topic_sum.resize(k);
  topic_sum.setConstant(1);
}
}  // namespace ml 
