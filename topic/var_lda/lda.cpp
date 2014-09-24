// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "lda.h"

#include <cstdio>
#include <cstdlib>
#include "base_head.h"

namespace ml {
void LdaSuffStats::Init(int m, int k, int v) {

  phi.resize(k);
  sum_phi.resize(k);

  for (int i = 0; i < k; i++) {
    phi[i].resize(v);
  }

  theta.resize(m);
  sum_theta.resize(m);

  for (int i = 0; i < m; i++) {
    theta[i].resize(k);
  }
}
}  // namespace ml 
