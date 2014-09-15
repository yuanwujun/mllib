// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "rtm_var_em.h"

#include <omp.h>
#include "base_head.h"
#include "eigen.h"
#include "eigen_util.h"
#include "rtm.h"

namespace ml {
double VarRTM::Likelihood(int d, RTMC &m, VecC &ga, Mat &phi) const {
  double g_sum = ga.sum();
  double digsum = DiGamma(g_sum);
  Vec expect(ga.size());
  for (int k = 0; k < ga.size(); k++) {
    expect[k] = DiGamma(ga[k]) - digsum;
  }
  double l = lgamma(m.alpha*ga.size()) - ga.size()*lgamma(m.alpha) - lgamma(g_sum);
  for (int k = 0; k < ga.size(); k++) {
    l += ((m.alpha - ga[k])*expect[k] + lgamma(ga[k]));
    for (size_t n = 0; n < cor.ULen(d); n++) {
      if (phi(k, n) > 0) {
        l += cor.Count(d, n) * phi(k, n) * (expect[k] - log(phi(k, n))
                              + m.ln_w(k, cor.Word(d, n)));
      }
    }
  }
  return l;
}

/****
phi: topic*doc_size
****/
void VarRTM::InitVar(int d, RTMC &m, Vec* digamma, Vec* ga, Mat* phi) const {
  ga->resize(m.TopicNum());
  digamma->resize(m.TopicNum());
  phi->resize(m.TopicNum(), cor.ULen(d));
  for (int k = 0; k < m.TopicNum(); k++) {
    (*ga)[k] = m.alpha + (cor.TLen(d) / ((double) m.TopicNum()));
    (*digamma)[k] = DiGamma((*ga)[k]);
  }
  phi->setConstant(1.0 / m.TopicNum());
}

/*****
Infer and compute suffstats, the motivation of infer is computing suffstats
phi: topic * doc_len
update z_bar
*****/
double VarRTM::EStep(int d, RTMC &m, Mat* z_bar, RTMSuffStats* ss) const {
  Mat phi(m.TopicNum(), cor.ULen(d)); 
  phi.setZero();
  Vec gamma;
  double likelihood = Infer(d, m, &gamma, &phi, z_bar);
  for (size_t n = 0; n < cor.ULen(d); n++) {
    for (int k = 0; k < m.TopicNum(); k++) {
      ss->topic(k, cor.Word(d, n)) += cor.Count(d, n) * phi(k, n);
      ss->topic_sum[k] += cor.Count(d, n) * phi(k, n);
    }
  }
  z_bar->col(d) = ZBar(d, phi);
  return likelihood;
}

Vec VarRTM::ZBar(int doc_id, MatC &phi) const {
  Vec v(phi.rows());
  v.setZero();
  for (int i = 0; i < phi.cols(); i++) {
    v += phi.col(i)*cor.Count(doc_id, i);
  }
  v /= cor.TLen(doc_id);
  return v;
}

void VarRTM::ZBar(VMatC &phi, Mat* m) const {
  m->resize(phi[0].rows(), phi.size());
  for (size_t i = 0; i < phi.size(); i++) {
    m->col(i) = ZBar(i, phi[i]);
  }
}

void VarRTM::RunEM(SpMat &test, RTM* m) {
  m->Init(cor.TermNum());
  RTMSuffStats ss;
  ss.InitSS(m->TopicNum(), m->TermNum());
  MStep(ss, m);
  m->alpha = initial_alpha_;
  double converged = 1;
  double likelihood_old = 0;
  Mat z_bar(m->TopicNum(), cor.Len());
  z_bar.setZero();
  for (int i = 0; i < em_max_iter_; i++) {
    double likelihood = 0;
    ss.SetZero(m->TopicNum(), m->TermNum());
    for (size_t d = 0; d < cor.Len(); d++) {
      likelihood += EStep(d, *m, &z_bar, &ss);
    }
    MStep(ss, m);
    MaxEta(z_bar, rho, m);
    converged = (likelihood_old - likelihood) / (likelihood_old);
    if (converged < 0) {
      var_max_iter_ = var_max_iter_ * 2;
    }
    likelihood_old = likelihood;
    LOG(INFO) << i << ":" << LinkPredict(test, *m, z_bar)
              << " " << likelihood;
  }
}

/****
phi: dimensions is doc_id, matrix is topic * doc_size 
z_bar: topic*doc_id
****/
double VarRTM::Infer(int d, RTMC &m, Vec* ga, Mat* phi, Mat* z_bar) const {
  double likelihood_old = 0;
  double c = 1;
  Vec digamma;
  InitVar(d, m, &digamma, ga, phi);
  for(int it = 1; (c > var_converged_) && (it < var_max_iter_); ++it) {
    Vec vec(m.TopicNum());
    vec.setZero();
    for (SpMatInIt it(net, d); it; ++it) {
      vec += z_bar->col(it.index());
    }
    Vec gradient = vec.cwiseProduct(m.eta);
    //for (size_t n = 0; n < cor.ULen(d); n++) {
    for (size_t n = 0; n < 1; n++) {
      for (int k = 0; k < m.TopicNum(); k++) {
        (*phi)(k, n) = digamma[k] + m.ln_w(k, cor.Word(d, n)) + gradient[k];
      }
      double ln_phi_sum = LogSum(phi->col(n));
      for (int k = 0; k < m.TopicNum(); k++) { //normalize phi
        (*phi)(k, n) = exp((*phi)(k, n) - ln_phi_sum);
      }
    }
    ga->setConstant(m.alpha);
    for (size_t n = 0; n < cor.ULen(d); n++) {
      for (int k = 0; k < m.TopicNum(); k++) {
        (*ga)[k] += cor.Count(d, n) * (*phi)(k, n);
        digamma[k] = DiGamma((*ga)[k]);
      }
    }
    double likelihood = Likelihood(d, m, *ga, *phi);
    assert(!isnan(likelihood));
    c = (likelihood_old - likelihood) / likelihood_old;
    likelihood_old = likelihood;
  }
  return likelihood_old;
}

Vec Regularization(int k) {
  Vec vec(k);
  vec.setConstant(1.0 / k);
  return vec.cwiseProduct(vec);
}

/****
z_bar:row for topic and col for doc
****/
void VarRTM::MaxEta(const Mat &z_bar, int rou, RTM* m) const {
  Vec pi_alpha = Regularization(m->TopicNum());
  Vec one(m->eta.size());
  one.setOnes();
  for (int i = 0; i < net.cols(); i++) {
    for (SpMatInIt it(net, i); it; ++it) {
      Vec pi = z_bar.col(i).cwiseProduct(z_bar.col(it.index()));
      m->eta -= lambda*((one - Sigmoid(m->eta.dot(pi))*pi -
                Sigmoid(m->eta.dot(pi_alpha))*pi_alpha));
    }
  }
}

void VarRTM::MStep(const RTMSuffStats &ss, RTM* m) {
  for (int k = 0; k < m->TopicNum(); k++) {
    for (int w = 0; w < m->TermNum(); w++) {
      if (ss.topic(k, w) > 0) {
        m->ln_w(k, w) = log(ss.topic(k, w)) - log(ss.topic_sum[k]);
      } else {
        m->ln_w(k, w) = -100;
      }
    }
  }
}

double VarRTM::LinkPredict(const SpMat &test, RTMC &m, Mat &z_bar) const {
  double rmse = 0;
  for (int i = 0;i < test.cols(); i++) {
    Mat phi(m.TopicNum(), cor.Len()); 
    phi.setZero();
    Vec gamma(m.TopicNum());
    Infer(i, m, &gamma, &phi, &z_bar);
    for (SpMatInIt it(test, i); it; ++it) {
      Vec p = z_bar.col(i).cwiseProduct(z_bar.col(it.index()));
      rmse += Square(Sigmoid(p.dot(m.eta)));
    }
  }
  return std::sqrt(rmse/ test.nonZeros());
}

void VarRTM::Load(StrC &net_path, StrC &cor_path) {
  ReadData(net_path, &net);
  cor.LoadData(cor_path);
}

//p->n sample number, p->l feature number
//topic num is alpha.size()
//neg_num, negative sampling number
void VarRTM::LiblinearInputData(VecC &alpha, int neg_num, problem* m) const {
/*
  p->n = net.nonZeros();
  p->l += neg_num;
  p->l = topic_num;
  std::vector<Vec> feature;
  for (int i = 0; i < net.cols(); i++) {
    for (SpMatInIt it(net, i); it; ++it) {
      feature.push_back(z_bar.col(i).cwiseProduct(z_bar.col(it.index())));
    }
  }
  for (size_t i = 0; i < feature.size(); i++) {
    for (int j = 0; j < feature[i].size(); j++) {
      m->x[i].index = j;
      m->x[i].value = feature[i][j];
    }
    m->y[i] = 1;
  }
  // add negative sample, regularization
  Vec a = alpha / alpha.sum();
  a = a.cwiseProduct(a);
  for (size_t i = feature.size(); i < p->n; i++) {
    for (int j = 0; j < topic_num; j++) {
      m->x[i].index = j;
      m->x[i].value = a[i][j];
    }
  }
  // add negative sample, sampling 
  for (size_t i = feature.size(); i < p->n; i++) {

  }
*/
}
} // namespace ml 
