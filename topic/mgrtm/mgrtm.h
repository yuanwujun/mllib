// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef TOPIC_MGRTM_H_
#define TOPIC_MGRTM_H_
#include "base_head.h"
#include "document.h"
#include "eigen.h"

namespace ml {
struct MGRTM { //model include the model parameter
  double g_alpha; // hyperparameter
  Vec l_alpha; // hyperparameter
  Mat g_ln_w;  // global topic-word distribution, col index word, row index topic
  VMat l_ln_w;  // local topic-word distribution
  Vec gamma;
  Vec pi;
  Vec g_u;  //global supervised parameter
  Mat l_u;  //local suppervised parameter
  
  inline void Init(int local_k, int local_num, int global_k, int v_size,
                   double gamma, double local_alpha, double global_alpha);
  
  int GTopicNum() const { return g_ln_w.rows(); }
  int LTopicNum2() const { return l_ln_w[0].rows(); }
  int LTopicNum1() const { return static_cast<int>(l_ln_w.size()); }
  int TermNum() const { return g_ln_w.cols(); }
};
typedef const MGRTM MGRTMC;
typedef MGRTM MGR;
typedef const MGR MGRC;

struct MGRTMSuffStats {
  inline void SetZeroZBar(int g_k, int l_k1, int l_k2, int doc_num);
  inline void SetZero(int g_k, int l_k1, int l_k2, int v, int doc_num);
  inline void InitSS(int g_k, int l_k1, int l_k2, int v, int doc_num);
  inline void CorpusInit(CorpusC &cor, MGRTMC &m);

  VMat l_topic;
  Mat l_topic_sum;
  Mat g_topic;
  Vec g_topic_sum;
  Vec pi;
  Mat g_z_bar;  // real ss need element-wise dot of two doc
  VMat l_z_bar;  // real ss need element-wise dot of two doc
  double doc_num;
};
typedef const MGRTMSuffStats MGRTMSuffStatsC;
typedef MGRTMSuffStats MGRSS;
typedef const MGRTMSuffStats MGRSSC;

void MGRTM::Init(int l_size1, int l_size2, int global_k, int v_size,
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
  pi.setZero();
  
  g_u.resize(global_k);
  l_u.resize(l_size2, l_size1);
}

void MGRTMSuffStats::SetZeroZBar(int g_k, int l_k1, int l_k2, int doc_num) {
  g_z_bar.resize(g_k, doc_num);
  g_z_bar.setZero();
  l_z_bar.resize(doc_num);
  for (int i = 0; i < doc_num; i++) {
    l_z_bar[i].resize(l_k2, l_k1);
    l_z_bar[i].setZero();
  }
}

void MGRTMSuffStats::SetZero(int g_k, int l_k1, int l_k2, int v, int doc_num) { 
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
  SetZeroZBar(g_k, l_k1, l_k2, doc_num);
}

void MGRTMSuffStats::CorpusInit(CorpusC &cor, MGRTMC &m) {
  g_topic.resize(m.GTopicNum(), m.TermNum());
  g_topic_sum.resize(m.GTopicNum());
  doc_num = cor.Len();
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
  for (int j = 0; j < m.LTopicNum1(); j++) {
    l_topic[j].resize(m.LTopicNum2(), m.TermNum());
    for (int k = 0; k < m.LTopicNum1(); k++) {
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

  SetZeroZBar(m.GTopicNum(), m.LTopicNum1(), m.LTopicNum2(), doc_num);
}
}  // namespace ml 
#endif // TOPIC_MGCTM_H_
