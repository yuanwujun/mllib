// copyright 2014 lijiankou. all rights reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef TOPIC_VAR_MGRTM_H_
#define TOPIC_VAR_MGRTM_H_
#include "base_head.h"
#include "cokus.h"
#include "eigen.h"
#include "mgrtm.h"
#include "converged.h"

#include "linear.h"

namespace ml {
struct MGRVar {
  VVMat l_z;
  VMat l_theta;
  Mat eta;

  VMat g_z;
  Mat g_theta;

  VVec delta;
  Mat omega;
  
  inline void Init(CorpusC &cor, MGRTMC &m);
};
typedef const MGRVar MGRVarC;

class VarMGRTM {
 public:
  void RunEM(MGRTM* m);
  void Load(StrC &net_path, StrC &cor_path, int times);
  void Init(ConvergedC &converged,int rho);
  double PredictAUC(SpMatC &test, MGRTMC &m, Mat &g_z_bar, MatC &eta,
                            VMatC &l_z_bar) const;
 private:
  void LearningEta(MatC &g_z_bar, VMatC &l_z_bar, Vec* g_u, Mat* l_u) const;
  void AddPi(VecC &pi, int &feature_index, feature_node* x_space,
                                                   int &dim_index) const;
 
  double Likelihood(DocC &doc, MGRVarC &ss, MGRTMC &m) const;
  double Infer(int d, MGRTMC &m, MatC &g_z, VMatC &l_z,
                                         MGRVar* para) const;
  double Infer(int d, MGRTMC &m, MGRVar* para, Mat* g_z_bar,
                                         VMat* l_z_bar) const;
  double EStep(MGRTMC &m, MGRSS* ss) const;
  void MStep(MGRSSC &ss, MGRTM* m);

  Corpus cor_;
  SpMat net_;
  SpMat held_out_net_;
  Converged converged_;
  int rho_;
};

inline void VarMGRTM::Init(ConvergedC &converged,int rho) {
  converged_ = converged;
  rho_ = rho;
}

/****
phi: topic*doc_size
****/
inline void MGRVar::Init(CorpusC &cor, MGRTMC &m) {
  g_theta.resize(m.GTopicNum(), cor.Len());

  l_theta.resize(cor.Len());
  g_z.resize(cor.Len());
  l_z.resize(cor.Len());
  
  omega.resize(2, cor.Len());
  delta.resize(cor.Len());
  eta.resize(m.LTopicNum1(), cor.Len());
  for (size_t d = 0; d < cor.Len(); d++) {
    //g_theta
    Vec v_g_theta(m.GTopicNum());
    double a = m.g_alpha + cor.TLen(d) / static_cast<double>(m.GTopicNum());
    v_g_theta.setConstant(a);
    g_theta.col(d) = v_g_theta;

    //g_z
    Mat doc_g_z;
    doc_g_z.resize(m.GTopicNum(), cor.ULen(d));
    doc_g_z.setConstant(1.0 / m.GTopicNum());
    g_z[d].swap(doc_g_z);

    //l_theta
    Mat doc_l_theta;
    doc_l_theta.resize(m.LTopicNum2(), m.LTopicNum1());
    a = m.l_alpha[0] + cor.TLen(d) / static_cast<double>(m.LTopicNum2());
    doc_l_theta.setConstant(a);
    l_theta[d].swap(doc_l_theta);
  
    //l_z
    VMat doc_l_z;
    doc_l_z.resize(m.LTopicNum1());
    for (size_t j = 0; j < doc_l_z.size(); j++) {
      doc_l_z[j].resize(m.LTopicNum2(), cor.ULen(d));
      doc_l_z[j].setConstant(1.0 / m.LTopicNum2());
    }
    l_z[d].swap(doc_l_z);

    //omega
    Vec doc_omega(2);
    doc_omega.setConstant(0.5); 
    omega.col(d) = doc_omega;

    //delta
    delta[d].resize(cor.ULen(d));
    delta[d].setConstant(0.5); 
  
    //eta
    Vec doc_eta(m.LTopicNum1());
    doc_eta.setConstant(1.0 / m.LTopicNum1());
    eta.col(d) = doc_eta;
  }
}
} // namespace ml
#endif // TOPIC_VAR_MGCTM_H_
