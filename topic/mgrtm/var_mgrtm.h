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

  Mat delta;
  Mat omega;
  
  void ReSet();
  void Init(CorpusC &cor, MGRTMC &m);
};
typedef const MGRVar MGRVarC;

class VarMGRTM {
 public:
  void RunEM(MGRTM* m);
  void Load(StrC &net_path, StrC &cor_path, int times);
  void Init(ConvergedC &converged);
  double PredictAUC(SpMat &test, MGRTMC &m, Mat &g_z_bar, VMatC &l_z_bar);
  double PredictAUC(SpMat &test, MGRTMC &m, Mat &g_z_bar, MatC &eta,
                            VMatC &l_z_bar);
 private:
  void LearningEta(MatC &g_z_bar, VMatC &l_z_bar, Vec* g_u, Mat* l_u) const;
  void AddPi(VecC &pi, int feature_index, feature_node* x_space) const;
  void LibLinearSample(MatC &g_z_bar, VMatC &l_z_bar,
                       feature_node* x_space, problem* prob) const;
 
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
};
} // namespace ml
#endif // TOPIC_VAR_MGCTM_H_
