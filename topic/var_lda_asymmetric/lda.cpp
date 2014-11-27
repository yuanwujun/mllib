// Copyright 2014 yuanwujun. All Rights Reserved.
// Author: real.yuanwj@gmail.com (WuJun Yuan)

#include <cstdio>
#include <cstdlib>

#include "lda.h"
#include "base_head.h"
#include "cokus.h"

namespace ml {
const int NUM_INIT = 1;

void LdaModel::InitModel(int k, int t, double* seed_alpha) {
  num_topics = k;
  num_terms = t;

  alpha = new double[num_topics];
  for(int i = 0; i<num_topics; ++i) {
    alpha[i] = seed_alpha[i];
  }

  log_prob_w = (double**)malloc(sizeof(double*)*num_topics);
  for (int i = 0; i < num_topics; i++) {
    log_prob_w[i] = (double*)malloc(sizeof(double)*num_terms);
    for (int j = 0; j < num_terms; j++) {
      log_prob_w[i][j] = 0.0;
    }
  }
}

LdaSuffStats::LdaSuffStats(LdaModel &m): class_word(NULL), class_total(NULL), alpha_suffstats(NULL), num_docs(0) {
  double init = 0.0;
  
  class_total = new double[m.num_topics];
  for (int i=0; i<m.num_topics; ++i) {
    class_total[i] = init;
  }


  class_word = (double**)malloc(sizeof(double*)*m.num_topics);
  for (int i=0; i<m.num_topics; i++) {
    class_word[i] = (double*)malloc(sizeof(double)*m.num_terms);
    for (int j=0; j<m.num_terms; j++) {
      class_word[i][j] = init;
    }
  }  
  alpha_suffstats = new double[m.num_topics];
  for (int i=0; i<m.num_topics; ++i) {
    alpha_suffstats[i] = init;
  }
}

void LdaSuffStats::RandomInitSS(const LdaModel &m) {
  for (int k = 0; k < m.num_topics; k++) {
    for (int n = 0; n < m.num_terms; n++) {
      class_word[k][n] += 1.0 / m.num_terms + myrand();
      class_total[k] += class_word[k][n];
    }
  }
}

void LdaSuffStats::CorpusInitSS(const Corpus &c, const LdaModel &m) {
  for (int k = 0; k < m.num_topics; k++) {
    for (int i = 0; i < NUM_INIT; i++) {
      const Document &doc = c.docs[static_cast<int>(floor(myrand() * c.docs.size()))];
      for (size_t n = 0; n < doc.words.size(); n++) {
        class_word[k][doc.words[n]] += doc.counts[n];
      }
    }

    for (int n = 0; n < m.num_terms; n++) {
      class_word[k][n] += 1.0;
      class_total[k] = class_total[k] + class_word[k][n];
    }
  }
}

void LdaSuffStats::InitSS(const LdaModel &model, double value) {
  for (int k = 0; k < model.num_topics; k++) {
    class_total[k] = value;
    for (int w = 0; w < model.num_terms; w++) {
      class_word[k][w] = value;
    }
  }
  num_docs = value;
  for (int k=0; k<model.num_topics; k++) {
    alpha_suffstats[k] = value;
  }
}
}  // namespace ml 
