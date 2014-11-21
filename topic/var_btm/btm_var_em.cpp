// Copyright 2014 yuanwujun. All Rights Reserved.
// Author: real.yuanwj@gmail.com (WuJun Yuan)
#include "btm_var_em.h"

#include "topic_util.h"

#include <omp.h>
#include "base_head.h"

namespace ml {
/*
double VarBTM::Likelihood(const Corpus &cor, int d, LdaModelC &m, VRealC &gamma,
                                                 VVRealC &phi) const {
  double g_sum = std::accumulate(gamma.begin(), gamma.end(), 0.0);
  double digsum = DiGamma(g_sum);
  const int &num = m.num_topics;
  VReal expect(num);
  for (int k = 0; k < num; k++) {
    expect.at(k) = DiGamma(gamma.at(k)) - digsum;
  }
  double l = lgamma(m.alpha * num) - num * lgamma(m.alpha) - lgamma(g_sum);
  for (int k = 0; k < num; k++) {
    l += ((m.alpha - gamma.at(k)) * expect[k] + lgamma(gamma.at(k)));
    for (size_t n = 0; n < cor.ULen(d); n++) {
      if (phi[n][k] > 0) {
        l += cor.Count(d, n) * phi[n][k] * (expect[k] - log(phi[n][k])
                              + m.log_prob_w[k][cor.Word(d, n)]);
      }
    }
  }
  return l;
}
*/

void VarBTM::InitVar(Mat& phi, Mat &z) const {

}

void VarBTM::RunEM() {
  for (int i = 0; i < em_max_iter_; i++) {
    // EStep
    for(int j = 1; j < var_max_iter_; ++j) {
      Mat phi_ep;
      LThetaEp(phi_var_, &phi_ep);
      int n = 0;
      for (int v = 0; v < phi_var_.cols(); ++v) {
        for (SpMatInIt it(biterm_net_, v); it; ++it) {
          for (int k = 0; k < topic_; ++k) {
            z_var_(k,n) = theta_(k) + 
              (phi_ep(k,it.row()) + phi_ep(k,it.col())) * it.value();
          }
          ++n;
        }
      }
      phi_var_.setConstant(beta_);
      for (size_t n = 0; n < biterms_.size(); ++n) {
        Triple &biterm = biterms_[n];
        for (int k = 0; k < topic_; ++k) {
          phi_var_(k,biterm.row()) += z_var_(k,n) * biterm.value();
          phi_var_(k,biterm.col()) += z_var_(k,n) * biterm.value();
        }
      }
    }
    // MStep
    for (int k = 0; k < topic_; ++k) {
      for (int n = 0; n < z_var_.cols(); ++n) {
        theta_(k) += z_var_(k,n);
      }
    }
  }
}

void VarBTM::Load(StrC &biterm_path) {
  std::pair<int,int> biterm_max_index = ReadData(biterm_path, &biterm_net_);
  for (int d = 0;d < biterm_net_.cols(); d++) {
    for (SpMatInIt it(biterm_net_, d); it; ++it) {
      int row = it.row();
      int col = it.col();
      int val = it.value();
      biterms_.push_back(Triple(row, col, val));
    }
  }
}
} // namespace ml
