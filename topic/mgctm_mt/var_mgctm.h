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
};
typedef const MGVar MGVarC;

class VarMGCTM {
 public:
  void RunEM(CorpusC &test, MGCTM* m);
  void Load(StrC &cor_path);
  void Init(ConvergedC &converged);
 private:
  double Likelihood(DocC &doc, MGVarC &ss, MGCTMC &m) const;
  double Infer(DocC &doc, MGCTMC &m, MGVar* var) const;
  double Infer(CorpusC &test, MGCTMC &m);
  double EStep(DocC &doc, MGCTMC &m, MGSS* ss, int iterate) const;
  void MStep(MGSSC &ss, MGCTM* m);
  void InitVar(DocC &doc, MGCTMC &m, MGVar* var)const;

  void OutputFile(MGCTMC& m,StrC &eta,int iterate) const;
 
  Corpus cor_;
  Converged converged_;
};
} // namespace ml
#endif // TOPIC_VAR_MGCTM_H_
