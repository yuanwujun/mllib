// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef ML_LDA_LDA_H_
#define ML_LDA_LDA_H_
#include "base_head.h"
#include "document.h"
namespace ml {
struct LdaModel {
  double alpha;  // hyperparameter
  double beta;   // hyperparameter
  double** log_prob_w;  // topic-word distribution
  VVReal theta;  // document-topic distribution
  VVReal phi;
  int num_topics;
  int num_terms;

  LdaModel() : alpha(0.01), log_prob_w(NULL), num_topics(0), num_terms(0) { }
};
typedef const LdaModel LdaModelC;

struct LdaSuffStats {
  VVInt phi;
  VVInt theta;
  VInt sum_phi;
  VInt sum_theta;

  double** class_word;
  double* class_total;
  double alpha_suffstats;
  int num_docs;
  LdaSuffStats() : class_word(NULL), class_total(NULL),
                    alpha_suffstats(0), num_docs(0) { }
  void Init(int m, int k, int v);
};
typedef const LdaSuffStats LdaSuffStatsC;

}  // namespace ml 
#endif  // ML_LDA_LDA_H_
