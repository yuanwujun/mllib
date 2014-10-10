// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "var_mgctm.h"

#include "eigen_util.h"
#include "mgctm.h"

namespace ml {
double VarMGCTM::Likelihood(DocC &doc, MGVarC &var, MGCTMC &m) const {
  double likelihood = 0;
  likelihood -= LogDelta(m.GTopicNum(), m.g_alpha);
  likelihood += LogDelta(var.g_theta);
  //1
  Vec g_theta_ep;
  GThetaEp(var.g_theta, &g_theta_ep); 
  for (int k = 0; k < g_theta_ep.size(); k++) {
    likelihood += (m.g_alpha - var.g_theta[k])*g_theta_ep[k];
  }

  //2
  Mat l_theta_ep;
  LThetaEp(var.l_theta, &l_theta_ep);
  for (int j = 0; j < m.LTopicNum1(); j++) {
    double a = 0;
    a -= LogDelta(m.LTopicNum2(), m.l_alpha[j]);
    a += LogDelta(var.l_theta.col(j));
    for (int k = 0; k < m.LTopicNum2(); k++) {
      a += (m.l_alpha[j] - var.l_theta(k, j))*l_theta_ep(k, j);
    }
    a *= var.eta[j];
    likelihood += a;
  }

  //3
  for (size_t n = 0; n < doc.ULen(); n++) {
    for (int k = 0; k < m.GTopicNum(); k++) {
      likelihood += doc.Count(n)*var.g_z(k, n)*(1 - var.delta(n))*
                    (g_theta_ep[k] - log(var.g_z(k, n)));
    }
  }

  //4
  for (int j = 0; j < m.LTopicNum1(); j++) {
    for (size_t n = 0; n < doc.ULen(); n++) {
      for (int k = 0; k < m.LTopicNum2(); k++) {
        likelihood += doc.Count(n)*var.l_z[j](k, n)*var.delta(n)*var.eta[j]*
                    (l_theta_ep(k, j) - log(var.l_z[j](k, n)));
      }
    }
  }

  //5
  likelihood -= LogBeta(m.gamma);
  likelihood += LogBeta(var.omega);

  Vec omega_ep;
  OmegaEp(var.omega, &omega_ep);
  likelihood += (m.gamma[1] - omega_ep[1])*omega_ep[1];
  likelihood += (m.gamma[0] - omega_ep[0])*omega_ep[0];

  //6
  for (size_t n = 0; n < doc.ULen(); n++) {
    likelihood += var.delta[n]*(omega_ep[1] - log(var.delta[n]));
    likelihood += (1 - var.delta[n])*(omega_ep[0] - log(1 - var.delta[n]));
  }

  //7
  for (int j = 0; j < m.LTopicNum1(); j++) {
    likelihood += var.eta[j]*(log(m.pi[j]) - var.eta[j]);
  }

  //8
  for (size_t n = 0; n < doc.ULen(); n++) {
    for (int k = 0; k < m.GTopicNum(); k++) {
      likelihood += doc.Count(n) * (1 - var.delta[n])*var.g_z(k, n)*
                    m.g_ln_w(k, doc.Word(n));
    }
  }

  for (int j = 0; j < m.LTopicNum1(); j++) {
    for (size_t n = 0; n < doc.ULen(); n++) {
      for (int k = 0; k < m.LTopicNum2(); k++) {
        likelihood += doc.Count(n) * var.delta[n]*var.l_z[j](k, n)*
                    m.l_ln_w[j](k, doc.Word(n));
      }
    }
  }
  return likelihood;
}

/****
phi: topic*doc_size
****/
void VarMGCTM::InitVar(DocC &doc, MGCTMC &m, Vec* g_diga, Mat* l_diga,
                                             MGVar* var) const {
  //g_theta
  var->g_theta.resize(m.GTopicNum());
  double a = m.g_alpha + doc.TLen() / static_cast<double>(m.GTopicNum());
  var->g_theta.setConstant(a);
  g_diga->resize(m.GTopicNum());
  g_diga->setConstant(DiGamma(a));

  //g_z
  var->g_z.resize(m.GTopicNum(), doc.ULen());
  var->g_z.setConstant(1.0 / m.GTopicNum());

  //l_theta
  var->l_theta.resize(m.LTopicNum2(), m.LTopicNum2());
  a = m.l_alpha[0] + doc.TLen() / static_cast<double>(m.LTopicNum2());
  var->l_theta.setConstant(a);
  l_diga->resize(m.LTopicNum2(), m.LTopicNum1());
  l_diga->setConstant(DiGamma(a));
  
  //l_z
  var->l_z.resize(m.LTopicNum1());
  for (size_t j = 0; j < var->l_z.size(); j++) {
    var->l_z[j].resize(m.LTopicNum2(), doc.ULen());
    var->l_z[j].setConstant(1.0 / m.LTopicNum2());
  }

  //omega
  var->omega.resize(2); 
  var->omega.setZero(); 

  //delta
  var->delta.resize(doc.ULen()); 
  var->delta.setZero(); 
  
  //eta
  var->eta.resize(m.LTopicNum1());
  var->eta.setZero();
}

/*****
Infer and compute suffstats, the motivation of infer is computing suffstats
phi: topic * doc_len
*****/
double VarMGCTM::EStep(DocC &doc, MGCTMC &m, MGSS* ss) const {
  MGVar var;
  double likelihood = Infer(doc, m, &var);
  for (int k = 0; k < m.GTopicNum(); k++) {
    for (size_t n = 0; n < doc.ULen(); n++) {
      ss->g_topic(k, doc.Word(n)) += doc.Count(n)*var.g_z(k, n);
      ss->g_topic_sum[k] += doc.Count(n) * var.g_z(k, n);
    }
  }
  for (int j = 0; j < m.LTopicNum1(); j++) {
    for (size_t n = 0; n < doc.ULen(); n++) {
      for (int k = 0; k < m.LTopicNum2(); k++) {
        ss->l_topic[j](k, doc.Word(n)) += doc.Count(n)*var.l_z[j](k, n);
        ss->l_topic_sum(k, j) += doc.Count(n)*var.l_z[j](k, n);
      }
    }
  }
  for (int j = 0; j < m.LTopicNum1(); j++) {
    ss->pi[j] += var.eta[j];
  }
  return likelihood;
}

void VarMGCTM::RunEM(MGCTM* m) {
  MGSS ss;
  ss.CorpusInit(cor_, *m);
  MStep(ss, m);
  LOG(INFO) << m->pi;
  LOG(INFO) << "start em";
  for (int i = 0; i < converged_.em_max_iter_; i++) {
    double likelihood = 0;
    ss.SetZero(m->GTopicNum(), m->LTopicNum1(), m->LTopicNum2(), m->TermNum());
    //for (size_t d = 0; d < cor_.Len(); d++) {
    for (size_t d = 0; d < 500; d++) {
      LOG(INFO) << i << " " << d << " " << likelihood;
      likelihood += EStep(cor_.docs[d], *m, &ss);
    }
    LOG(INFO) << likelihood;
    MStep(ss, m);
    VMGVar var;
    //double per = Infer(test_, *m, &var);
    //LOG(INFO) << per;
  }
}

double VarMGCTM::Infer(DocC &doc, MGCTMC &m, MGVar* para) const {
  double c = 1;
  Vec g_diga;
  Mat l_diga;
  InitVar(doc, m, &g_diga, &l_diga, para);
  MGVar &p = *para;
  for(int it = 1; (c > converged_.var_converged_) && (it < 
                       converged_.var_max_iter_); ++it) {
    //delta
    p.omega[0] = m.gamma[0];
    for (size_t n = 0; n < doc.ULen(); n++) {
      p.omega[0] += p.delta(n);
    }

    p.omega[1] = m.gamma[1];
    for (size_t n = 0; n < doc.ULen(); n++) {
      p.omega[1] += (1 - p.delta(n));
    }

    //global theta and global z
    for (size_t n = 0; n < doc.ULen(); n++) {
      for (int k = 0; k < m.GTopicNum(); k++) {
        p.g_z(k, n) = (1 - p.delta[n])*g_diga[k] + m.g_ln_w(k, doc.Word(n));
      }
      double ln_phi_sum = LogSum(p.g_z.col(n));
      for (int k = 0; k < m.GTopicNum(); k++) { //normalize g_z
        p.g_z(k, n) = exp(p.g_z(k, n) - ln_phi_sum);
      }
    }

    p.g_theta.setConstant(m.g_alpha);
    for (size_t n = 0; n < doc.ULen(); n++) {
      for (int k = 0; k < m.GTopicNum(); k++) {
        p.g_theta[k] += doc.Count(n) * p.g_z(k, n);
      }
    }
    for (int k = 0; k < m.GTopicNum(); k++) {
      g_diga[k] = DiGamma(p.g_theta[k]);
    }

    //local theta and local z
    for (size_t n = 0; n < doc.ULen(); n++) {
      for (int j = 0; j < m.LTopicNum1(); j++) {
        for (int k = 0; k < m.LTopicNum2(); k++) {
          p.l_z[j](k, n) = p.delta[n]*p.eta[j]*l_diga(k, j) +
                           m.l_ln_w[j](k, doc.Word(n));
        }
      }

      Vec ln_local_z_var_sum(m.LTopicNum1());
      for (int j = 0; j < m.LTopicNum1(); j++) {
        ln_local_z_var_sum[j] = LogSum(p.l_z[j].col(n));
      }
      for (int j = 0; j < m.LTopicNum1(); j++) { //normalize l_z
        for (int k = 0; k < m.LTopicNum2(); k++) {
          p.l_z[j](k, n) = exp(p.l_z[j](k, n) - ln_local_z_var_sum[j]);
        }
      }
    }

    for (int j = 0; j < m.LTopicNum1(); j++) {
      for (int k = 0; k < m.LTopicNum2(); k++) {
        p.l_theta(k, j) = m.l_alpha[j];
      }
      for (size_t n = 0; n < doc.ULen(); n++) {
        for (int k = 0; k < m.LTopicNum2(); k++) {
          p.l_theta(k, j) += doc.Count(n) * p.l_z[j](k, n);
        }
      }
    }

    for (int j = 0; j < m.LTopicNum1(); j++) {
      for (int k = 0; k < m.LTopicNum2(); k++) {
        l_diga(k, j) = DiGamma(p.l_theta(k, j));
      }
    }

    Mat l_theta_ep;
    LThetaEp(p.l_theta, &l_theta_ep);
    Vec g_theta_ep;
    GThetaEp(p.g_theta, &g_theta_ep); 
    //indicate variable eta
    for (int j = 0; j < m.LTopicNum1(); j++) {
      p.eta[j] = log(m.pi[j]);
      int l_topic_num = m.LTopicNum2();
      p.eta[j] += lgamma(l_topic_num*m.l_alpha[j]);
      p.eta[j] -= l_topic_num*lgamma(m.LTopicNum2());
      for (int k = 0; k < m.LTopicNum2(); k++) {
        p.eta[j] += (m.l_alpha[j] - 1)*l_theta_ep(k, j);
      }

      for (size_t n = 0; n < doc.ULen(); n++) {
        double a = 0;
        for (int k = 0; k < m.LTopicNum2(); k++) {
          a += p.l_z[j](k, n) * l_theta_ep(k, j);
          a += m.l_ln_w[j](k, doc.Word(n));
        }
        p.eta[j] += p.delta[n]*a;
      }
    }
    double ln_eta_sum = LogSum(p.eta);
    for (int k = 0; k < m.LTopicNum2(); k++) { //normalize eta
      p.eta(k) = exp(p.eta(k) - ln_eta_sum);
    }

    //variable delta
    for (size_t n = 0; n < doc.ULen(); n++) {
      double a = 0;
      a -= DiGamma(m.gamma[0]);
      a += DiGamma(m.gamma[1]);

      for (int j = 0; j < m.LTopicNum1(); j++) {
        for (int k = 0; k < m.LTopicNum2(); k++) {
          a -= p.eta[j]*p.l_z[j](k, n)*(l_theta_ep(k, j)); 
        }
      }

      for (int j = 0; j < m.LTopicNum1(); j++) {
        for (int k = 0; k < m.LTopicNum2(); k++) {
          a -= p.eta[j]*p.l_z[j](k, n)*m.l_ln_w[j](k, doc.Word(n));
        }
      }

      for (int k = 0; k < m.LTopicNum2(); k++) {
        a += p.g_z(k, n) * g_theta_ep[k];
      }

      for (int k = 0; k < m.LTopicNum2(); k++) {
        a += p.g_z(k, n) * m.g_ln_w(k, doc.Word(n));
      }
      p.delta[n] = Sigmoid(-a);
    }
  }
  return Likelihood(doc, *para, m);
}

void VarMGCTM::MStep(MGSSC &ss, MGCTM* m) {
  //local
  for (int j = 0; j < m->LTopicNum1(); j++) {
    m->pi[j] = ss.pi[j] / ss.doc_num;
    for (int k = 0; k < m->LTopicNum2(); k++) {
      for (int w = 0; w < m->TermNum(); w++) {
        if (ss.l_topic[j](k, w) > 0) {
          m->l_ln_w[j](k, w) = log(ss.l_topic[j](k, w)) -
                               log(ss.l_topic_sum(j, k));
        } else {
          m->l_ln_w[j](k, w) = -100;
        }
      }
    }
  }

  //global
  for (int k = 0; k < m->GTopicNum(); k++) {
    for (int w = 0; w < m->TermNum(); w++) {
      if (ss.g_topic(k, w) > 0) {
        m->g_ln_w(k, w) = log(ss.g_topic(k, w)) - log(ss.g_topic_sum[k]);
      } else {
        m->g_ln_w(k, w) = -100;
      }
    }
  }
}

void VarMGCTM::Load(StrC &cor_path) {
  Corpus corpus;
  corpus.LoadData(cor_path);
  SplitData(corpus, 0.8, &cor_, &test_);
}

void VarMGCTM::Init(ConvergedC &converged) {
  converged_ = converged;
}

double VarMGCTM::Infer(CorpusC &cor, MGC &m, VMGVar* var) const {
  var->resize(cor.Len());
  double sum = 0;
  for (size_t i = 0; i < cor.Len(); i++) {
    LOG(INFO) << "infer:" << i; 
    Vec g_diga;
    Mat l_diga;
    InitVar(cor.docs[i], m, &g_diga, &l_diga, &(var->at(i)));
    double l = Infer(cor.docs[i], m, &(var->at(i)));
    sum += l;
  }
  return exp(- sum / cor.TWordsNum());
}
} // namespace ml 
