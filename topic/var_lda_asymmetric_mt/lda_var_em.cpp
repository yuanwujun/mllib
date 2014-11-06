// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include <omp.h>

#include "array_numeric_operator.h"

#include "lda_var_em.h"
#include "base_head.h"
#include "lda.h"

namespace ml {
void LDA::Init(float em_converged, int em_max_iter, int estimate_alpha,int var_max_iter, int var_converged) {
  em_converged_ = em_converged;
  em_max_iter_ = em_max_iter;
  estimate_alpha_ = estimate_alpha;
  var_converged_ = var_converged;
  var_max_iter_ = var_max_iter;
}

double LDA::Likelihood(const Corpus &cor, int d, const LdaModel &m, VRealC &gamma,VVRealC &phi) const {
  double alpha_sum = double_array_sum(m.alpha,m.num_topics);
  double gamma_sum = std::accumulate(gamma.begin(), gamma.end(), 0.0);
  double digsum = DiGamma(gamma_sum);
  const int &num = m.num_topics;
  VReal expect(num);
  for (int k = 0; k < num; k++) {
    expect.at(k) = DiGamma(gamma.at(k)) - digsum;
  }

  double l = lgamma(alpha_sum) - lgamma(gamma_sum);
  
  for (int k = 0; k < num; k++) {
    l += ((m.alpha[k] - gamma.at(k)) * expect[k] + lgamma(gamma.at(k)) - lgamma(m.alpha[k]));
    for (size_t n = 0; n < cor.ULen(d); n++) {
      if (phi[n][k] > 0) {
        l += cor.Count(d, n) * phi[n][k] * (expect[k] - log(phi[n][k]) + m.log_prob_w[k][cor.Word(d, n)]);
      }
    }
  }
  return l;
}

void LDA::InitVarParamter(const Corpus &cor, int d, const LdaModel &m, VReal* digamma,VReal* ga, VVReal* phi) const {
  ga->resize(m.num_topics);
  digamma->resize(m.num_topics);
  phi->resize(cor.ULen(d));
  for (int k = 0; k < m.num_topics; k++) {
    (*ga)[k] = m.alpha[k] + (cor.docs[d].total / ((double) m.num_topics));
    (*digamma)[k] = DiGamma((*ga)[k]);
  }
  for (VReal::size_type n = 0; n < phi->size(); n++) {
    phi->at(n).resize(m.num_topics);
    for (int k = 0; k < m.num_topics; k++) {
      (*phi)[n][k] = 1.0 / m.num_topics;
    }
  }
}

double LDA::Infer(CorpusC &cor, int d, const LdaModel &m, VReal* ga, VVReal* phi) const {
  VReal digamma(m.num_topics);
  double likelihood_old = 0;
  double c = 1;
  InitVarParamter(cor, d, m, &digamma, ga, phi);
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

    for (size_t i = 0; i < ga->size(); i++) {
      ga->at(i) = m.alpha[i];
    }
    
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

void LDA::RunEM(const Str &type, CorpusC &train, CorpusC &test, LdaModel* m) {
  LdaSuffStats ss(*m);
  if (type == "seeded") {
    ss.CorpusInitSS(train, *m);
  } else if (type == "random") {
    ss.RandomInitSS(*m);
  }

  LdaMLE(0, ss, m);
  double converged = 1;
  double likelihood_old = 0;
  for (int i = 0; i < em_max_iter_; i++) {
    VVReal gamma(train.Len());
    VVVReal phi(train.Len());
    VReal likelihood(train.Len());
    #pragma omp parallel for
    for (size_t d = 0; d < train.Len(); d++) {
      likelihood[d] = Infer(train, d, *m, &gamma[d], &phi[d]);
    }

    double likelihoods = 0;
    ss.InitSS(*m, 0);
    for (size_t d = 0; d<train.Len(); d++) {
      double gamma_sum = 0;
      for (int k = 0; k < m->num_topics; k++) {
        gamma_sum += gamma[d][k];
      }
      for (int k = 0; k < m->num_topics; k++) {
        ss.alpha_suffstats[k] += DiGamma(gamma[d][k]) - DiGamma(gamma_sum);
      }

      for (size_t n = 0; n < train.ULen(d); n++) {
        for (int k = 0; k < m->num_topics; k++) {
          ss.class_word[k][train.Word(d, n)] += train.Count(d, n) * phi[d][n][k];
          ss.class_total[k] += train.Count(d, n) * phi[d][n][k];
        }
      }
      ss.num_docs = ss.num_docs + 1;
      likelihoods += likelihood[d];
    }

    LdaMLE(estimate_alpha_, ss, m);
    converged = (likelihood_old - likelihoods) / (likelihood_old);
    if (converged < 0) {
      var_max_iter_ = var_max_iter_ * 2;
    }
    likelihood_old = likelihoods;

    if (i % 10 == 0) {
      VVReal gamma2;
      VVVReal phi2;
      LOG(INFO) << "em " << i << " perplexity:" << Infer(test, *m, &gamma2, &phi2);
    }
  }
}

double LDA::Infer(CorpusC &cor, const LdaModel &m,VVReal* ga, VVVReal* phi) const {
  ga->resize(cor.Len());
  phi->resize(cor.Len());
  VReal likelihood(cor.Len());
  
  #pragma omp parallel for
  for (size_t d = 0; d < cor.Len(); d++) {
    likelihood[d] = Infer(cor, d, m, &(ga->at(d)), &(phi->at(d)));
  }

  double sum = 0.0;
  for (size_t d = 0; d < cor.Len(); d++) {
    sum += likelihood[d];
  }
  return exp(- sum / cor.TWordsNum());
}
} // namespace ml 
