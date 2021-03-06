// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "var_mgctm.h"

#include "eigen_util.h"
#include "mgctm.h"
#include "topic_util.h"

namespace ml {
double VarMGCTM::Likelihood(DocC &doc, MGVarC &var, MGCTMC &m) const {
  double likelihood = 0;
  likelihood -= LogDelta(m.GTopicNum(), m.g_alpha);
  likelihood += LogDelta(var.g_theta);
  Vec g_theta_ep;
  GThetaEp(var.g_theta, &g_theta_ep); 
  for (int k = 0; k < g_theta_ep.size(); k++) {
    likelihood += (m.g_alpha - var.g_theta[k])*g_theta_ep[k];
  }

  Mat l_theta_ep;
  LThetaEp(var.l_theta, &l_theta_ep);
  for (int j = 0; j < m.LTopicNum1(); j++) {
    double a = 0;
    a -= LogDelta(m.LTopicNum2(), m.l_alpha[j]);
    a += LogDelta(var.l_theta.col(j));
    for (int k = 0; k < m.LTopicNum2(); k++) {
      a += (m.l_alpha[j] - var.l_theta(k, j))*l_theta_ep(k, j);
    }
    likelihood += a*var.eta[j];
  }

  for (size_t n = 0; n < doc.ULen(); n++) {
    for (int k = 0; k < m.GTopicNum(); k++) {
      likelihood += doc.Count(n)*var.g_z(k, n)*(1 - var.delta(n))*
                    (g_theta_ep[k] - log(var.g_z(k, n)));
    }
  }

  for (int j = 0; j < m.LTopicNum1(); j++) {
    for (size_t n = 0; n < doc.ULen(); n++) {
      for (int k = 0; k < m.LTopicNum2(); k++) {
        likelihood += doc.Count(n)*var.l_z[j](k, n)*var.delta(n)*var.eta[j]*
                    (l_theta_ep(k, j) - log(var.l_z[j](k, n)));
      }
    }
  }

  likelihood -= LogBeta(m.gamma);
  likelihood += LogBeta(var.omega);

  Vec omega_ep;
  OmegaEp(var.omega, &omega_ep);
  likelihood += (m.gamma[1] - var.omega[1])*omega_ep[1];
  likelihood += (m.gamma[0] - var.omega[0])*omega_ep[0];

  for (size_t n = 0; n < doc.ULen(); n++) {
    double a = var.delta[n];
    likelihood += doc.Count(n)*(a*omega_ep[1] - log(pow(a, a)));
    a = 1 - var.delta[n];
    likelihood += doc.Count(n)*(a*omega_ep[0] - log(pow(a, a)));
  }

  for (int j = 0; j < m.LTopicNum1(); j++) {
    likelihood += var.eta[j]*(log(m.pi[j]) - var.eta[j]);
  }

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
                    m.l_ln_w[j](k, doc.Word(n)) * var.eta[j];
      }
    }
  }
  return likelihood;
}

/****
phi: topic*doc_size
****/
void VarMGCTM::InitVar(DocC &doc, MGCTMC &m, MGVar* var) const {
  //g_theta
  var->g_theta.resize(m.GTopicNum());
  var->g_theta.setConstant(m.g_alpha + doc.TLen()/(double)m.GTopicNum());

  //g_z
  var->g_z.resize(m.GTopicNum(), doc.ULen());
  var->g_z.setConstant(1.0 / m.GTopicNum());

  //l_theta
  var->l_theta.resize(m.LTopicNum2(), m.LTopicNum1());
  var->l_theta.setConstant(m.l_alpha[0] +
                 doc.TLen()/(double)m.LTopicNum2());
  
  //l_z
  var->l_z.resize(m.LTopicNum1());
  for (size_t j = 0; j < var->l_z.size(); j++) {
    var->l_z[j].resize(m.LTopicNum2(), doc.ULen());
    var->l_z[j].setConstant(1.0 / m.LTopicNum2());
  }

  double frac = m.GTopicNum() / (double)(m.LTopicNum1()*m.LTopicNum2() + m.GTopicNum());

  //omega
  var->omega.resize(2); 
  var->omega[0] = m.gamma[0] + doc.TLen()*frac;
  var->omega[1] = m.gamma[1] + doc.TLen()*(1 - frac);

  //delta
  var->delta.resize(doc.ULen()); 
  var->delta.setConstant(1 - frac); 
  
  //eta
  var->eta.resize(m.LTopicNum1());
  var->eta.setConstant(1.0 / m.LTopicNum1());
}

void VarMGCTM::RunEM(CorpusC &test, MGCTM* m) {
  MGSS ss;
  ss.CorpusInit(cor_, *m);
  MStep(ss, m);
  LOG(INFO) << m->pi.transpose();
  for (int i = 0; i < converged_.em_max_iter_; i++) {
    std::vector<MGVar> vars(cor_.Len());
    VReal likelihoods(cor_.Len());
    #pragma omp parallel for
    for (size_t d = 0; d < cor_.Len(); d++) {
      likelihoods[d] = Infer(cor_.docs[d], *m, &vars[d]);
    }

    double likelihood = 0;
    VStr etas(cor_.Len());
    ss.SetZero(m->GTopicNum(), m->LTopicNum1(), m->LTopicNum2(), m->TermNum());
    for (size_t d = 0; d < cor_.Len(); d++) {
      DocC &doc = cor_.docs[d];
      for (size_t n = 0; n < doc.ULen(); n++) {
        for (int k = 0; k < m->GTopicNum(); k++) {
          ss.g_topic(k, doc.Word(n)) += doc.Count(n)*vars[d].g_z(k, n)*
                                     (1 - vars[d].delta[n]);
          ss.g_topic_sum[k] += doc.Count(n)*vars[d].g_z(k, n)*(1 - vars[d].delta[n]);
        }
      }
      for (int j = 0; j < m->LTopicNum1(); j++) {
        for (size_t n = 0; n < doc.ULen(); n++) {
          for (int k = 0; k < m->LTopicNum2(); k++) {
            ss.l_topic[j](k, doc.Word(n)) += doc.Count(n)*vars[d].l_z[j](k, n)
                                *vars[d].delta[n]*vars[d].eta[j];
            ss.l_topic_sum(k, j) += doc.Count(n)*vars[d].l_z[j](k, n) *
                                  vars[d].delta[n] * vars[d].eta[j];
          }
        }
      }
      for (int j = 0; j < m->LTopicNum1(); j++) {
        ss.pi[j] += vars[d].eta[j];
      }

      etas[d] = EVecToStr(vars[d].eta);
      likelihood += likelihoods[d];
    }
    MStep(ss, m);
    LOG(INFO) << m->pi.transpose();
    OutputFile(*m, Join(etas,"\n"), i);
//    LOG(INFO) <<"perplexity: " <<Infer(test,*m);
  }
}

double VarMGCTM::Infer(CorpusC &test, MGCTMC &m) {
  double sum = 0.0;
  VReal likelihoods(test.Len());
  #pragma omp parallel for
  for (size_t d = 0; d < test.Len(); d++) {
    MGVar var;
    likelihoods[d] = Infer(test.docs[d], m, &var);
  }
  for (size_t d = 0; d < test.Len(); d++) {
    sum += likelihoods[d];
  }
  return exp(- sum / test.TWordsNum());
}

double VarMGCTM::Infer(DocC &doc, MGCTMC &m, MGVar* para) const {
  double c = 1;
  InitVar(doc, m, para);
  MGVar &p = *para;
  for(int it = 1; (c > converged_.var_converged_) && (it < 
                       converged_.var_max_iter_); ++it) {
    //indicate variable eta
    Vec g_theta_ep;
    GThetaEp(p.g_theta, &g_theta_ep);
    Mat l_theta_ep;
    LThetaEp(p.l_theta, &l_theta_ep);
    for (int j = 0; j < m.LTopicNum1(); j++) {
      p.eta[j] = log(m.pi[j]);
      int l_topic_num = m.LTopicNum2();
      p.eta[j] += lgamma(l_topic_num*m.l_alpha[j]);
      p.eta[j] -= l_topic_num*lgamma(m.l_alpha[j]);
      for (int k = 0; k < m.LTopicNum2(); k++) {
        p.eta[j] += (m.l_alpha[j] - 1)*l_theta_ep(k, j);
      }

      for (size_t n = 0; n < doc.ULen(); n++) {
        double a = 0;
        for (int k = 0; k < m.LTopicNum2(); k++) {
          a += p.l_z[j](k, n) * l_theta_ep(k, j);
          a += p.l_z[j](k, n) * m.l_ln_w[j](k, doc.Word(n));
        }
        p.eta[j] += p.delta[n]*a*doc.Count(n);
      }
    }

    double ln_eta_sum = LogSum(p.eta);
    for (int j = 0; j < m.LTopicNum1(); j++) { //normalize eta
      p.eta[j] = exp(p.eta[j] - ln_eta_sum);
    }

    //omega
    p.omega[1] = m.gamma[1];
    for (size_t n = 0; n < doc.ULen(); n++) {
      p.omega[1] += p.delta(n)*doc.Count(n);
    }

    p.omega[0] = m.gamma[0];
    for (size_t n = 0; n < doc.ULen(); n++) {
      p.omega[0] += (1 - p.delta(n))*doc.Count(n);
    }

    //local theta
    for (int j = 0; j < m.LTopicNum1(); j++) {
      for (int k = 0; k < m.LTopicNum2(); k++) {
        p.l_theta(k, j) = p.eta[j] * m.l_alpha[j];
      }
    }
    for (int j = 0; j < m.LTopicNum1(); j++) {
      for (int k = 0; k < m.LTopicNum2(); k++) {
        for (size_t n = 0; n < doc.ULen(); n++) {
          p.l_theta(k, j) += doc.Count(n)* p.delta[n]*p.eta[j]* p.l_z[j](k, n);
        }
        p.l_theta(k, j) += 1 - p.eta[j];
      }
    }

    //global theta
    p.g_theta.setConstant(m.g_alpha);
    for (size_t n = 0; n < doc.ULen(); n++) {
      for (int k = 0; k < m.GTopicNum(); k++) {
        p.g_theta[k] += doc.Count(n) * p.g_z(k, n) * (1 - p.delta[n]);
      }
    }
 
    for (size_t n = 0; n < doc.ULen(); n++) {
      //variable delta
      double tmp = DiGamma(m.gamma[1]) - DiGamma(m.gamma[0]);
      for (int j = 0; j < m.LTopicNum1(); j++) {
        for (int k = 0; k < m.LTopicNum2(); k++) {
          tmp += p.eta[j]*p.l_z[j](k, n)*l_theta_ep(k,j); 
          tmp += p.eta[j]*p.l_z[j](k, n)*m.l_ln_w[j](k, doc.Word(n));
        }
      }
      for (int k = 0; k < m.GTopicNum(); k++) {
        tmp -= p.g_z(k, n) * g_theta_ep[k];
        tmp -= p.g_z(k, n) * m.g_ln_w(k, doc.Word(n));
      }
      p.delta[n] = Sigmoid(tmp);

      //local z
      for (int j = 0; j < m.LTopicNum1(); j++) {
        for (int k = 0; k < m.LTopicNum2(); k++) {
          p.l_z[j](k, n) = p.delta[n]*p.eta[j]*(l_theta_ep(k, j) +
                           m.l_ln_w[j](k, doc.Word(n)));
        }
        double ln_local_z_sum = LogSum(p.l_z[j].col(n));
        for (int k = 0; k < m.LTopicNum2(); k++) {
          p.l_z[j](k, n) = exp(p.l_z[j](k, n) - ln_local_z_sum);
        }
      }

      //global z
      for (int k = 0; k < m.GTopicNum(); k++) {
        p.g_z(k, n) = (1 - p.delta[n])*(g_theta_ep[k] + m.g_ln_w(k, doc.Word(n)));
      }
      double ln_z_sum = LogSum(p.g_z.col(n));
      for (int k = 0; k < m.GTopicNum(); k++) { //normalize g_z
        p.g_z(k, n) = exp(p.g_z(k, n) - ln_z_sum);
      }
    }

  }
  return Likelihood(doc, p, m);
}

void VarMGCTM::MStep(MGSSC &ss, MGCTM* m) {
  //local
  #pragma omp parallel for
  for (int j = 0; j < m->LTopicNum1(); j++) {
    //m->pi[j] = ss.pi[j] / ss.doc_num;
    m->pi[j] = ss.pi[j] / ss.pi.sum();
    for (int k = 0; k < m->LTopicNum2(); k++) {
      for (int w = 0; w < m->TermNum(); w++) {
        if (ss.l_topic[j](k, w) > 0) {
          m->l_ln_w[j](k, w) = log(ss.l_topic[j](k, w)) -
                               log(ss.l_topic_sum(k, j));
        } else {
          m->l_ln_w[j](k, w) = -100;
        }
      }
    }
  }

  //global
  #pragma omp parallel for
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
  cor_.LoadData(cor_path);
}

void VarMGCTM::Init(ConvergedC &converged) {
  converged_ = converged;
}

void VarMGCTM::OutputFile(MGCTMC& m,StrC &eta,int iterate) const {
  Str eta_file = "/data0/data/ctm/model/eta_" + ToStr(iterate);
  WriteStrToFile(eta,eta_file);

  VStr gLnBeta(m.GTopicNum());
  for (int row = 0; row < m.g_ln_w.rows(); ++row) {
    gLnBeta[row] = EVecToStr(m.g_ln_w.row(row)); 
  }
  Str global_beta = "/data0/data/ctm/model/gbeta_" + ToStr(iterate);
  WriteStrToFile(Join(gLnBeta,"\n"),global_beta);

  Str local_beta = "/data0/data/ctm/model/lbeta_" + ToStr(iterate) +"_";
  for (size_t group = 0; group < m.l_ln_w.size(); ++group) {
    VStr lLnBeta(m.LTopicNum2());
    for (int row = 0; row < m.l_ln_w[group].rows(); ++row) {
      lLnBeta[row] = EVecToStr(m.l_ln_w[group].row(row));
    }
    Str local_beta_group = local_beta + ToStr(group);
    WriteStrToFile(Join(lLnBeta,"\n"),local_beta_group);
  }
}
} // namespace ml 
