// copyright 2013 lijiankou. all rights reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef ML_LDA_VAR_EM_H
#define ML_LDA_VAR_EM_H
#include "base_head.h"

#include "lda.h"
#include "lda_model.h"
#include "cokus.h"

namespace ml {
class LDA {
 public:
  double Likelihood(const Corpus &cor, int d, const LdaModel &m, VRealC &gamma,VVRealC &phi) const;
  double Infer(const Corpus &cor, int d, const LdaModel &m, VReal* ga,VVReal* phi) const;

  double Infer(const LdaModel &m, VVReal* ga, VVVReal* phi) const;
  double Infer(CorpusC &cor, const LdaModel &m, VVReal* ga, VVVReal* phi) const;
 
  void RunEM(const Str &type, CorpusC &train, CorpusC &test, LdaModel* m);

  void Init(float em_converged, int em_max_iter, int estimate_alpha,int var_max_iter, int var_converged);
 private:
  void InitVarParamter(const Corpus &cor, int d, const LdaModel &m, VReal* digamma,VReal* ga, VVReal* phi) const;

  double DocEStep(const Corpus &cor, int d, const LdaModel &m,LdaSuffStats* ss) const;
  float em_converged_;
  int em_max_iter_;
  int estimate_alpha_;
  int var_max_iter_;
  int var_converged_;
};

} // namespace ml
#endif // ML_LDA_VAR_EM_H
