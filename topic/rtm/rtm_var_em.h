// copyright 2014 lijiankou. all rights reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef ML_TOPIC_RTM_VAR_EM_H
#define ML_TOPIC_RTM_VAR_EM_H
#include "base_head.h"

#include "eigen.h"
#include "rtm.h"
#include "cokus.h"

namespace ml {
class VarRTM {
 public:
  double Likelihood(int d, RTMC &m, VecC &ga, Mat &phi) const;
  //double Infer(int d, RTMC &m, Vec* ga, Mat* phi, Mat* z_bar) const;
  
  double Infer(int d, RTMC &m, MatC &z_bar, Vec* ga, Mat* phi) const;
 
  inline void Init(float em_converged, int em_max_iter, int estimate_alpha,
                   int var_max_iter, int var_converged_,
                   double initial_alpha, int n_topic);
  void MStep(const RTMSuffStats &suff, RTM* m);
  void MaxEta(const Mat &z_bar, int rho, RTM* m) const;
  void LiblinearInputData(VecC &alpha, int neg_num, problem* m) const;
  Vec ZBar(int doc_id, MatC &phi) const;
  double LinkPredict(const SpMat &test, RTMC &rtm, Mat &z_bar) const;
  void RunEM(SpMat &test, RTM* m);
  void Load(StrC &net_path, StrC &cor_path);
 private:
  void InitVar(int d, RTMC &m, Vec* digamma, Vec* ga, Mat* phi) const;

  double EStep(int d, RTMC &m, Mat* z_bar, RTMSuffStats* ss) const;
  float em_converged_;
  int em_max_iter_;
  int estimate_alpha_;
  double initial_alpha_;
  int var_max_iter_;
  int n_topic_;
  int var_converged_;
  int rho;
  Corpus cor;
  SpMat net;
  double lambda;
};

void VarRTM::Init(float em_converged, int em_max_iter, int estimate_alpha,
                                   int var_max_iter, int var_converged,
                                   double initial_alpha, int n_topic) {
  em_converged_ = em_converged;
  em_max_iter_ = em_max_iter;
  estimate_alpha_ = estimate_alpha;
  initial_alpha_ = initial_alpha;
  n_topic_ = n_topic;
  var_converged_ = var_converged;
  var_max_iter_ = var_max_iter;
}
} // namespace ml
#endif // ML_TOPIC_RTM_VAR_EM_H
