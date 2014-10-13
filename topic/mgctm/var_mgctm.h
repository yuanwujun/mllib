// copyright 2014 lijiankou. all rights reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef TOPIC_VAR_MGCTM_H_
#define TOPIC_VAR_MGCTM_H_
#include "base_head.h"
#include "cokus.h"
#include "eigen.h"
#include "mgctm.h"
#include "converged.h"

namespace ml {
struct MGVar {
  VMat l_z;
  Mat l_theta;
  Vec eta;

  Mat g_z;
  Vec g_theta;

  Vec delta;
  Vec omega;

  void ReSet();
  void Init();
};
typedef const MGVar MGVarC;

class VarMGCTM {
 public:
  void RunEM(MGCTM* m);
  void Load(StrC &cor_path);
  void Init(ConvergedC &converged);
 private:
  double Likelihood(DocC &doc, MGVarC &ss, MGCTMC &m) const;
  double Infer(DocC &doc, MGCTMC &m, MGVar* var) const;
  double EStep(DocC &doc, MGCTMC &m, MGSS* ss) const;
  void MStep(MGSSC &ss, MGCTM* m);
  void InitVar(DocC &doc, MGCTMC &m, Vec* g_diga, Mat* l_diga, MGVar* var)const;
  inline void GThetaEp(VecC &vec, Vec* des) const;
  inline void LThetaEp(MatC &mat, Mat* des) const;
  inline void OmegaEp(VecC &vec, Vec* des) const;
  inline double LogDelta(VecC &vec) const;
  inline double LogDelta(int num, double alpha) const;
  inline double LogBeta(VecC &vec) const;
 
  Corpus cor_;
  Converged converged_;
};

void VarMGCTM::GThetaEp(VecC &vec, Vec* des) const {
  double digsum = DiGamma(vec.sum());
  des->resize(vec.size());
  for (int k = 0; k < vec.size(); k++) {
    (*des)[k] = DiGamma(vec[k]) - digsum;
  }
}

void VarMGCTM::LThetaEp(MatC &mat, Mat* des) const {
  des->resize(mat.rows(), mat.cols());
  for (int j = 0; j < mat.cols(); j++) {
    double digsum = DiGamma(mat.col(j).sum());
    for (int k = 0; k < mat.rows(); k++) {
      (*des)(k, j) = DiGamma(mat(k, j)) - digsum;
    }
  }
}

void VarMGCTM::OmegaEp(VecC &vec, Vec* des) const {
  des->resize(vec.size());
  double sum = DiGamma(vec.sum());
  (*des)[0] = DiGamma(vec[0]) - sum;
  (*des)[1] = DiGamma(vec[1]) - sum;
}

double VarMGCTM::LogDelta(VecC &alpha) const {
  double res = 0;
  for (int i = 0; i < alpha.size(); i++) {
    res += lgamma(alpha[i]);
  }
  return res - lgamma(alpha.sum()); 
}

double VarMGCTM::LogBeta(VecC &vec) const {
  return lgamma(vec.sum()) - lgamma(vec[0]) - lgamma(vec[1]);
}

double VarMGCTM::LogDelta(int num, double alpha) const {
  return num * lgamma(alpha) - lgamma(num * alpha);
}
} // namespace ml
#endif // TOPIC_VAR_MGCTM_H_
