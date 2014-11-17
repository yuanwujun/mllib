// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef TOPIC_MGCTM_H_
#define TOPIC_MGCTM_H_
#include "base_head.h"
#include "cokus.h"
#include "document.h"
#include "eigen.h"

namespace ml {
struct MGCTM { //model include the model parameter
  double g_alpha; // hyperparameter
  Vec l_alpha; // hyperparameter
  Mat g_ln_w;  // global topic-word distribution, col index word, row index topic
  VMat l_ln_w;  // local topic-word distribution
  Vec gamma;
  Vec pi;
  
  inline void Init(int local_k, int local_num, int global_k, int v_size,
                   double gamma, double local_alpha, double global_alpha);
  
  int GTopicNum() const { return g_ln_w.rows(); }
  int LTopicNum2() const { return l_ln_w[0].rows(); }
  int LTopicNum1() const { return static_cast<int>(l_ln_w.size()); }
  int TermNum() const { return g_ln_w.cols(); }
};
typedef const MGCTM MGCTMC;
typedef MGCTM MG;
typedef const MG MGC;

void MGCTM::Init(int l_size2, int l_size1, int global_k, int v_size,
                 double ga, double local_alpha, double global_alpha) {
  g_alpha = global_alpha;
  l_alpha.resize(l_size1);
  l_alpha.setConstant(local_alpha);

  g_ln_w.resize(global_k, v_size);
  g_ln_w.setZero();
  l_ln_w.resize(l_size1);
  for (size_t i = 0; i < l_ln_w.size(); i++) {
    l_ln_w[i].resize(l_size2, v_size);
    l_ln_w[i].setZero();
  }

  gamma.resize(2);
  gamma.setConstant(ga);

  pi.resize(l_size1);
  //pi.setConstant(1.0 / l_size1);
}

struct MGCTMSuffStats {
  inline void SetZero(int g_k, int l_k1, int l_k2, int v);
  inline void CorpusInit(CorpusC &cor, MGCTMC &m);

  VMat l_topic;
  Mat l_topic_sum;
  Mat g_topic;
  Vec g_topic_sum;
  Vec pi;
  double doc_num;
};
typedef const MGCTMSuffStats MGCTMSuffStatsC;
typedef MGCTMSuffStats MGSS;
typedef const MGCTMSuffStats MGSSC;

void MGCTMSuffStats::SetZero(int g_k, int l_k1, int l_k2, int v) { 
  g_topic.resize(g_k, v);
  g_topic.setZero();

  g_topic_sum.resize(g_k);
  g_topic_sum.setZero();

  l_topic.resize(l_k1);
  for (size_t i = 0; i < l_topic.size(); i++) {
    l_topic[i].resize(l_k2, v);
    l_topic[i].setZero();
  }
  l_topic_sum.resize(l_k2, l_k1);
  l_topic_sum.setZero();

  pi.setZero();
}

void MGCTMSuffStats::CorpusInit(CorpusC &cor, MGCTMC &m) {
  doc_num = cor.Len();
  g_topic.resize(m.GTopicNum(), m.TermNum());
  g_topic.setZero();
  g_topic_sum.resize(m.GTopicNum());
  g_topic_sum.setZero();
  for (int k = 0; k < m.GTopicNum(); k++) {
    for (int i = 0; i < 1; i++) {
      const Document &doc = 
        cor.docs[static_cast<int>(floor(myrand() * cor.Len()))];
      for (size_t n = 0; n < doc.words.size(); n++) {
        g_topic(k, doc.words[n]) += doc.Count(n);
      }
    }
    for (int n = 0; n < m.TermNum(); n++) {
      g_topic(k, n) += 1.0;
      g_topic_sum[k] += g_topic(k, n);
    }
  }

  l_topic.resize(m.LTopicNum1());
  l_topic_sum.resize(m.LTopicNum2(), m.LTopicNum1());
  l_topic_sum.setZero();
  for (int j = 0; j < m.LTopicNum1(); j++) {
    l_topic[j].resize(m.LTopicNum2(), m.TermNum());
    l_topic[j].setZero();
    for (int k = 0; k < l_topic[j].rows(); k++) {
      for (int i = 0; i < 1; i++) {
        const Document &doc =
              cor.docs[static_cast<int>(floor(myrand()*cor.Len()))];
        for (size_t n = 0; n < doc.words.size(); n++) {
          l_topic[j](k, doc.words[n]) += doc.Count(n);
        }
      }
      for (int n = 0; n < m.TermNum(); n++) {
        l_topic[j](k, n) += 1.0;
        l_topic_sum(k, j) += l_topic[j](k, n);
      }
    }
  }
  
  pi.resize(m.LTopicNum1());
  pi.setConstant(cor.Len() / m.LTopicNum1());
}
}  // namespace ml 
#endif // TOPIC_MGCTM_H_
