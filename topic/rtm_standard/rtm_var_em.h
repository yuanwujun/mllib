// copyright 2014 lijiankou. all rights reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef ML_TOPIC_RTM_VAR_EM_H
#define ML_TOPIC_RTM_VAR_EM_H
#include "base_head.h"

#include "eigen.h"
#include "rtm.h"
#include "cokus.h"

namespace ml {
inline Vec ZBar(DocC &doc, MatC &phi) {
  Vec v(phi.rows());
  v.setZero();
  for (int n = 0; n < phi.cols(); n++) {
    v += phi.col(n)* doc.Count(n);
  }
  v /= doc.TLen();
  return v;
}

struct RTMVar {
  Mat gamma; //k*m k is the topic number, m is the doc number
  VMat phi; //doc*k*n, doc num, topic num, and doc length
  Mat z_bar; // k*m, topic num and doc num

  inline void Init(CorpusC &cor, RTMC &m);
};
typedef const RTMVar RTMVarC;

/****
phi: doc_num*topic_num*doc_length 
****/
void RTMVar::Init(CorpusC &cor, RTMC &m) {
  gamma.resize(m.TopicNum(), cor.Len());
  phi.resize(cor.Len());
  for (size_t d = 0; d < phi.size(); d++) {
    phi[d].resize(m.TopicNum(), cor.ULen(d));
    phi[d].setConstant(1.0 / m.TopicNum());
    for (int k = 0; k < m.TopicNum(); k++) {
      gamma(k, d) = m.alpha + (cor.TLen(d) / ((double) m.TopicNum()));
    }
  }
  z_bar.resize(m.TopicNum(), cor.Len());
  for (size_t d = 0; d < cor.Len(); d++) {
    z_bar.col(d) = ZBar(cor.docs[d], phi[d]);
  }
}

class VarRTM {
 public:
  void Infer(CorpusC &cor, RTMC &m, RTMVar* var) const;
 
  inline void Init(float em_converged, int em_max_iter, int estimate_alpha,
                   int var_max_iter,int doc_var_max_iter , int var_converged_,
                   double initial_alpha, int n_topic,int rho);
  void EStep(CorpusC &cor, RTMC &m, RTMSuffStats* ss) const;
  void MStep(const RTMSuffStats &suff, RTM* m);
  void LearningEta(VecC &alpha, const Mat &z_bar, Vec *eta) const;
  void RunEM(RTM* m);
  void Load(StrC &net_path, StrC &cor_path, int times);
  double PredictAUC(RTMC &m, Mat &z_bar) ;
 private:
  void InitVar(int d, RTMC &m, Vec* digamma, Vec* ga, Mat* phi) const;

  float em_converged_;
  int em_max_iter_;
  int estimate_alpha_;
  double initial_alpha_;
  int var_max_iter_;
  int doc_var_max_iter_;
  int n_topic_;
  int var_converged_;
  int rho_;
  Corpus cor;
  SpMat net;
  SpMat held_out_net_;
};

void VarRTM::Init(float em_converged, int em_max_iter, int estimate_alpha,
                                   int var_max_iter,int doc_var_max_iter, int var_converged,
                                   double initial_alpha, int n_topic,int rho) {
  em_converged_ = em_converged;
  em_max_iter_ = em_max_iter;
  estimate_alpha_ = estimate_alpha;
  initial_alpha_ = initial_alpha;
  n_topic_ = n_topic;
  var_converged_ = var_converged;
  var_max_iter_ = var_max_iter;
  doc_var_max_iter_ = doc_var_max_iter;
  rho_ = rho;
}
} // namespace ml
#endif // ML_TOPIC_RTM_VAR_EM_H
