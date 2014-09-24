// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "lda_var_em.h"

#include <omp.h>
#include "base_head.h"
#include "lda.h"

namespace ml {

void LDA::CreateSS(StrC &t, CorpusC &c, LdaModelC &m, LdaSuffStats* ss) const {
  if (t == "seeded") {
    CorpusInitSS(c, m, ss);
  } else if (t == "random") {
    RandomInitSS(m, ss);
  }
}

double LDA::Likelihood(const Corpus &cor, int d, LdaModelC &m, VRealC &gamma,
                                                 VVRealC &phi) const {
  double g_sum = std::accumulate(gamma.begin(), gamma.end(), 0.0);
  double digsum = DiGamma(g_sum);
  const int &num = m.num_topics;
  VReal expect(num);
  for (int k = 0; k < num; k++) {
    expect.at(k) = DiGamma(gamma.at(k)) - digsum;
  }
  double l = lgamma(m.alpha * num) - num * lgamma(m.alpha) - lgamma(g_sum);
  for (int k = 0; k < num; k++) {
    l += ((m.alpha - gamma.at(k)) * expect[k] + lgamma(gamma.at(k)));
    for (size_t n = 0; n < cor.ULen(d); n++) {
      if (phi[n][k] > 0) {
        l += cor.Count(d, n) * phi[n][k] * (expect[k] - log(phi[n][k])
                              + m.log_prob_w[k][cor.Word(d, n)]);
      }
    }
  }
  return l;
}

void LDA::InitVar(const Corpus &cor, int d, LdaModelC &m, VReal* digamma,
                                      VReal* ga, VVReal* phi) const {
  ga->resize(m.num_topics);
  digamma->resize(m.num_topics);
  phi->resize(cor.ULen(d));
  for (int k = 0; k < m.num_topics; k++) {
    (*ga)[k] = m.alpha + (cor.docs[d].total / ((double) m.num_topics));
    (*digamma)[k] = DiGamma((*ga)[k]);
  }
  for (VReal::size_type n = 0; n < phi->size(); n++) {
    phi->at(n).resize(m.num_topics);
    for (int k = 0; k < m.num_topics; k++) {
      (*phi)[n][k] = 1.0 / m.num_topics;
    }
  }
}

/*****
Infer and compute suffstats, the motivation of infer is computing suffstats
***/

double LDA::DocEStep(const Corpus &cor, int d, LdaModelC &m,
                     LdaSuffStats* ss) const {
  VReal gamma;
  VVReal phi;
  double likelihood = Infer(cor, d, m, &gamma, &phi);
  double gamma_sum = 0;
  for (int k = 0; k < m.num_topics; k++) {
    gamma_sum += gamma[k];
    ss->alpha_suffstats += DiGamma(gamma[k]);
  }
  ss->alpha_suffstats -= m.num_topics * DiGamma(gamma_sum);
  for (size_t n = 0; n < cor.ULen(d); n++) {
    for (int k = 0; k < m.num_topics; k++) {
      ss->class_word[k][cor.Word(d, n)] += cor.Count(d, n) * phi[n][k];
      ss->class_total[k] += cor.Count(d, n) * phi[n][k];
    }
  }
  ss->num_docs = ss->num_docs + 1;
  return likelihood;
}
 
void LDA::RunEM(const Str &type, CorpusC &train, CorpusC &test, LdaModel* m) {
  NewLdaModel(n_topic_, train.num_terms, m);
  LdaSuffStats ss;
  NewLdaSuffStats(*m, &ss);
  CreateSS(type, train, *m, &ss);
  LdaMLE(0, ss, m);
  m->alpha = initial_alpha_;
  double converged = 1;
  double likelihood_old = 0;
  for (int i = 0; i < em_max_iter_; i++) {
    LOG(INFO) << i << " " << em_max_iter_ - i;
    double likelihood = 0;
    InitSS(*m, 0, &ss);
    for (size_t d = 0; d < train.Len(); d++) {
      likelihood += DocEStep(train, d, *m, &ss);
    }
    LdaMLE(estimate_alpha_, ss, m);
    converged = (likelihood_old - likelihood) / (likelihood_old);
    if (converged < 0) {
      var_max_iter_ = var_max_iter_ * 2;
    }
    likelihood_old = likelihood;

    VVReal gamma2;
    VVVReal phi2;
    LOG(INFO) << "em " << i << " perplexity:" << Infer(test, *m, &gamma2, &phi2);
  }
}

double LDA::Infer(CorpusC &cor, LdaModelC &m, VVReal* ga, VVVReal* phi) const {
  double sum = 0;
  ga->resize(cor.Len());
  phi->resize(cor.Len());
  for (size_t i = 0; i < cor.Len(); i++) {
    double l = Infer(cor, i, m, &(ga->at(i)), &(phi->at(i)));
    sum += l;
  }
  return exp(- sum / cor.TWordsNum());
}

double LDA::Infer(CorpusC &cor, int d, LdaModelC &m, VReal* ga,
                                            VVReal* phi) const {
  VReal digamma(m.num_topics);
  double likelihood_old = 0;
  double c = 1;
  InitVar(cor, d, m, &digamma, ga, phi);
  for(int it = 1; (c > var_converged_) && (it < var_max_iter_); ++it) {
    for (size_t n = 0; n < cor.ULen(d); n++) {
      for (int k = 0; k < m.num_topics; k++) {
        (*phi)[n][k] = digamma[k] + m.log_prob_w[k][cor.Word(d, n)];
      }
      double log_phi_sum = LogPartition(phi->at(n));
      for (int k = 0; k < m.num_topics; k++) {
        (*phi)[n][k] = exp((*phi)[n][k] - log_phi_sum);
      }
    }
    InitGamma(m.alpha, ga); 
    for (size_t n = 0; n < cor.ULen(d); n++) {
      for (int k = 0; k < m.num_topics; k++) {
        (*ga)[k] += cor.Count(d, n) * (*phi)[n][k];
        digamma[k] = DiGamma(ga->at(k));
      }
    }
    double likelihood = Likelihood(cor, d, m, *ga, *phi);
    assert(!isnan(likelihood));
    c = (likelihood_old - likelihood) / likelihood_old;
    likelihood_old = likelihood;
  }
  return likelihood_old;
}

/*
double LDA::Infer(CorpusC &cor, int d, LdaModelC &m, VReal* ga,
                                            VVReal* phi) const {
  VReal digamma(m.num_topics);
  InitVar(cor, d, m, &digamma, ga, phi);
  double likelihood_old = 0;
  double c = 1;
  for(int it = 1; (c > var_converged_) && (it < var_max_iter_); ++it) {
    for (size_t n = 0; n < cor.ULen(d); n++) {
      double phisum = 0;
      VReal oldphi(m.num_topics);
      for (int k = 0; k < m.num_topics; k++) {
        oldphi[k] = (*phi)[n][k];
        (*phi)[n][k] = digamma[k] + m.log_prob_w[k][cor.Word(d, n)];
        if (k > 0) {
          phisum = LogSum(phisum, (*phi)[n][k]);
        } else {
          phisum = (*phi)[n][k]; 
        }
      }
      for (int k = 0; k < m.num_topics; k++) {
        (*phi)[n][k] = exp((*phi)[n][k] - phisum);
        (*ga)[k] = (*ga)[k] + cor.Count(d, n) * ((*phi)[n][k] - oldphi[k]);
        digamma[k] = DiGamma((*ga)[k]);
      }
    }
    double likelihood = Likelihood(cor, d, m, *ga, *phi);
    assert(!isnan(likelihood));
    c = (likelihood_old - likelihood) / likelihood_old;
    likelihood_old = likelihood;
  }
  return likelihood_old;
}
*/

void InitGamma(double alpha, VReal* ga) {
  for (size_t i = 0; i < ga->size(); i++) {
    ga->at(i) = alpha;
  }
}
} // namespace ml 
