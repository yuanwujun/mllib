// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef ML_LDA_LDA_H_
#define ML_LDA_LDA_H_
#include "base_head.h"
#include "document.h"

namespace ml {
struct LdaModel {
  double* alpha;                    // hyperparameter
  double** log_prob_w;              // topic-word distribution
  int num_topics;                   // topic's number
  int num_terms;                    // word's number in the vocabulary

  LdaModel() : alpha(NULL), log_prob_w(NULL), num_topics(0), num_terms(0)  { }
  void InitModel(int k, int t, double* seed_alpha); 
};

struct LdaSuffStats {
  double** class_word;
  double* class_total;
  double* alpha_suffstats;
  int num_docs;
  
  LdaSuffStats(LdaModel &m);

  void RandomInitSS(const LdaModel &m);
  void CorpusInitSS(const Corpus &c, const LdaModel &m);
  void InitSS(const LdaModel &model, double value);
};

}  // namespace ml 
#endif  // ML_LDA_LDA_H_
