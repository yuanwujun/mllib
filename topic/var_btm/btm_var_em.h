// Copyright 2014 yuanwujun. All Rights Reserved.
// Author: real.yuanwj@gmail.com (WuJun Yuan)
#ifndef ML_BTM_VAR_EM_H_
#define ML_BTM_VAR_EM_H_
#include "base_head.h"
#include "document.h"
#include "cokus.h"
#include "eigen.h"

namespace ml {
class VarBTM {
 public:
  void RunEM();
  inline void Init(int em_max_iter,
               int var_max_iter,int n_topic,double beta);
  void Load(StrC &biterm_path);

 private:
  void InitVar(Mat& phi, Mat &z) const;

  int em_max_iter_;
  int var_max_iter_;
  int topic_;

  double beta_;          // hyperparameter
  EVec theta_;           // topic proportion
  Mat phi_var_;          // k*v k is the topic number, m is the term number
  Mat z_var_;            // k*n topic num and term-pair num
  SpMat biterm_net_;
  TripleVec biterms_;
};

void VarBTM::Init(int em_max_iter,int var_max_iter,int n_topic,double beta) {
  em_max_iter_ = em_max_iter;
  var_max_iter_ = var_max_iter;
  topic_ = n_topic;
  beta_ = beta;

  theta_.resize(topic_);
}
}  // namespace ml
#endif  // ML_BTM_VAR_EM_H_
