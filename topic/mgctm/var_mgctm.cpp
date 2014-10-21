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
double VarMGCTM::EStep(DocC &doc, MGCTMC &m, MGSS* ss, int iterate) const {
  MGVar var;
  double likelihood = Infer(doc, m, &var);
  for (size_t n = 0; n < doc.ULen(); n++) {
    for (int k = 0; k < m.GTopicNum(); k++) {
      ss->g_topic(k, doc.Word(n)) += doc.Count(n) * var.g_z(k, n) *
                                     (1 - var.delta[n]);
      ss->g_topic_sum[k] += doc.Count(n) * var.g_z(k, n) * (1 - var.delta[n]);
    }
  }
  for (int j = 0; j < m.LTopicNum1(); j++) {
    for (size_t n = 0; n < doc.ULen(); n++) {
      for (int k = 0; k < m.LTopicNum2(); k++) {
        ss->l_topic[j](k, doc.Word(n)) += doc.Count(n) * var.l_z[j](k, n)
                                * var.delta[n] * var.eta[j];
        ss->l_topic_sum(k, j) += doc.Count(n) * var.l_z[j](k, n) *
                                  var.delta[n] * var.eta[j];
      }
    }
  }
  for (int j = 0; j < m.LTopicNum1(); j++) {
    ss->pi[j] += var.eta[j];
  }

  DumpVarParamter(var,iterate);

  return likelihood;
}

void VarMGCTM::RunEM(CorpusC &test, MGCTM* m) {
  MGSS ss;
  ss.CorpusInit(cor_, *m);
  LOG(INFO) << "ss init over";
  MStep(ss, m);
  LOG(INFO) << "start em";
  for (int i = 0; i < converged_.em_max_iter_; i++) {
    double likelihood = 0;
    ss.SetZero(m->GTopicNum(), m->LTopicNum1(), m->LTopicNum2(), m->TermNum());
    for (size_t d = 0; d < cor_.Len(); d++) {
      likelihood += EStep(cor_.docs[d], *m, &ss, i);
      LOG_IF(INFO, d % 10 == 0) << d << " " << likelihood << " "
                                 << exp(- likelihood / cor_.TWordsNum());
    }
    MStep(ss, m);
    DumpModelParamter(*m,i);
    LOG(INFO) << likelihood << " " << exp(- likelihood / cor_.TWordsNum());
  }
}

double VarMGCTM::Infer(DocC &doc, MGCTMC &m, MGVar* para) const {
  double c = 1;
  Vec g_diga;
  Mat l_diga;
  InitVar(doc, m, &g_diga, &l_diga, para);
  MGVar &p = *para;
  Vec vec(m.GTopicNum());  //allociate one time
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
          p.l_theta(k, j) += doc.Count(n)* p.delta[n]*p.eta[j]* p.l_z[j](k, n)
                             + 1 - p.eta[j];
        }
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

  LOG(INFO) << "local over";
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
  LOG(INFO) << "global over";
}

void VarMGCTM::Load(StrC &cor_path) {
  cor_.LoadData(cor_path);
}

void VarMGCTM::Init(ConvergedC &converged) {
  converged_ = converged;
}

void VarMGCTM::DumpVarParamter(MGVarC& var, int iterate) const{
  Str eta_file = "model/eta" + ToStr(iterate);
  AppendStrToFile(EVecToStr(var.eta), eta_file);

  Str local_theta_file = "model/ltheta" + ToStr(iterate);
  for (int col = 0; col < var.l_theta.cols(); ++col) {
    AppendStrToFile(EVecToStr(var.l_theta.col(col)), local_theta_file); 
  }

  Str global_theta_file = "model/gtheta" + ToStr(iterate);
  AppendStrToFile(EVecToStr(var.g_theta), global_theta_file);
}

void VarMGCTM::DumpModelParamter(MGCTMC& m,int iterate) const {
  Str global_beta = "model/gbeta" + ToStr(iterate);
  for (int col = 0; col < m.g_ln_w.cols(); ++col) {
    AppendStrToFile(EVecToStr(m.g_ln_w.col(col)), global_beta);
  }

  Str local_beta = "model/lbeta" + ToStr(iterate);
  for (size_t group = 0; group < m.l_ln_w.size(); ++group) {
    Str local_beta_group = local_beta + ToStr(group);
    for (int col = 0; col < m.l_ln_w[group].cols(); ++col) {
      AppendStrToFile(EVecToStr(m.l_ln_w[group].col(col)), local_beta_group);
    }
  }
}
} // namespace ml 
