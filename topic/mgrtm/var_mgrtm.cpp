// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "var_mgrtm.h"

#include "eigen_util.h"
#include "topic_util.h"
#include "mgrtm.h"
#include "roc.h"

namespace ml {
Vec Mean(DocC &doc, MatC &g_z, VecC &delta) {
  Vec v(g_z.rows());
  v.setZero();
  for (int n = 0; n < g_z.cols(); n++) {
    v += doc.Count(n)*(1 - delta[n])*g_z.col(n);
  }
  return v / doc.TLen();
}

//l_z_bar doc
void Mean(DocC &doc, VMatC &l_z, VecC &delta, Mat* l_z_bar) {
  for (size_t j = 0; j < l_z.size(); j++) {
    Vec vec(l_z[0].rows());
    vec.setZero();
    for (int n = 0; n < l_z[0].cols(); n++) {
      vec += delta[n]*l_z[j].col(n)*doc.Count(n); 
    }
    l_z_bar->col(j) = vec / doc.TLen();
  }
}

//normalize, source is in the ln space
void Normalize(Vec* p) {
  double ln_sum = LogSum(*p);
  for (int j = 0; j < p->size(); j++) {
    (*p)[j] = exp((*p)[j] - ln_sum);
  }
}

//normalize, source is in the ln space
void NormalizeCol(int n, Mat* m) {
  double ln_sum = LogSum(m->col(n));
  for (int k = 0; k < m->rows(); k++) {
    (*m)(k, n) = exp((*m)(k, n) - ln_sum);
  }
}

double VarMGRTM::Likelihood(DocC &doc, MGRVarC &var, MGRTMC &m) const {
  double likelihood = 0;
  /*
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
  */
  return likelihood;
}

/*****
Infer and compute suffstats, the motivation of infer is computing suffstats
phi: topic * doc_len
*****/
double VarMGRTM::EStep(MGRTMC &m, MGRSS* ss) const {
  double likelihood = 0;
  MGRVar var;
  var.Init(cor_, m);
  for (size_t d = 0; d < cor_.Len(); d++) {
    likelihood += Infer(d, m, ss->g_z_bar, ss->l_z_bar, &var);

    ss->g_z_bar.col(d) = Mean(cor_.docs[d], var.g_z[d], var.delta[d]);
    Mean(cor_.docs[d], var.l_z[d], var.delta[d], &(ss->l_z_bar[d]));

    LOG_IF(INFO, d % 10 == 0) << d << " " << likelihood << " "
                                 << exp(- likelihood / cor_.TWordsNum());

    DocC &doc = cor_.docs[d];
    for (size_t n = 0; n < doc.ULen(); n++) {
      for (int k = 0; k < m.GTopicNum(); k++) {
        ss->g_topic(k, doc.Word(n)) += doc.Count(n)*var.g_z[d](k, n)*
                                       (1 - var.delta[d][n]);
        ss->g_topic_sum[k] += doc.Count(n)*var.g_z[d](k, n)*(1 - var.delta[d][n]);
      }
    }
    for (int j = 0; j < m.LTopicNum1(); j++) {
      for (size_t n = 0; n < doc.ULen(); n++) {
        for (int k = 0; k < m.LTopicNum2(); k++) {
          double a = doc.Count(n)*var.l_z[d][j](k, n)*
                                  var.delta[d][n]*var.eta(j, d);
          ss->l_topic[j](k, doc.Word(n)) += a;
          ss->l_topic_sum(k, j) += a;
        }
      }
    }
    for (int j = 0; j < m.LTopicNum1(); j++) {
      ss->pi[j] += var.eta(j, d);
    }
  }
  LOG(INFO) << PredictAUC(held_out_net_, m, ss->g_z_bar, var.eta, ss->l_z_bar);
  return likelihood;
}

void VarMGRTM::RunEM(MGRTM* m) {
  MGRSS ss;
  ss.CorpusInit(cor_, *m);
  LOG(INFO) << "ss init over";
  MStep(ss, m);
  LOG(INFO) << "start em";
  for (int i = 0; i < converged_.em_max_iter_; i++) {
    LOG(INFO) << i;
    double likelihood = 0;
    ss.SetZero(m->GTopicNum(), m->LTopicNum1(), m->LTopicNum2(),
                                                m->TermNum(), cor_.Len());
    likelihood += EStep(*m, &ss);
    MStep(ss, m);
    //LOG(INFO) << likelihood << " " << exp(- likelihood / cor_.TWordsNum());
  }
}

double Active(VecC &g_u, MatC &l_u, MatC &g, VMatC &l, MatC &eta,
                                             int d1, int d2) {
  double s = g_u.dot(g.col(d1).cwiseProduct(g.col(d2)));
  for (int j = 0; j < eta.rows(); j++) {
    s += eta(j, d1)*eta(j, d2)*l_u.col(j).dot(l[d1].col(j).cwiseProduct(l[d2].col(j)));
  }
  return Sigmoid(s);
}
 
double VarMGRTM::Infer(int d, MGRTMC &m, MatC &g_z_bar, VMatC &l_z_bar,
                                         MGRVar* para) const {
  double c = 1;
  MGRVar &p = *para;
  DocC &doc = cor_.docs[d];
  for(int it = 1; (c > converged_.var_converged_) && (it < 
                       converged_.var_max_iter_); ++it) {
    //indicate variable eta
    Vec g_theta_ep;
    GThetaEp(p.g_theta.col(d), &g_theta_ep);
    Mat l_theta_ep;
    LThetaEp(p.l_theta[d], &l_theta_ep);
    for (int j = 0; j < m.LTopicNum1(); j++) {
      p.eta(j, d) = log(m.pi[j]);
      int l_topic_num = m.LTopicNum2();
      p.eta(j, d) += lgamma(l_topic_num*m.l_alpha[j]);
      p.eta(j, d) -= l_topic_num*lgamma(m.l_alpha[j]);
      for (int k = 0; k < m.LTopicNum2(); k++) {
        p.eta(j, d) += (m.l_alpha[j] - 1)*l_theta_ep(k, j);
      }

      for (size_t n = 0; n < doc.ULen(); n++) {
        double a = 0;
        for (int k = 0; k < m.LTopicNum2(); k++) {
          a += p.l_z[d][j](k, n) * l_theta_ep(k, j);
          a += p.l_z[d][j](k, n) * m.l_ln_w[j](k, doc.Word(n));
        }
        p.eta(j, d) += p.delta[d][n]*a*doc.Count(n);
      }

      for (SpMatInIt it(net_, d); it; ++it) {
        VMatC &l = l_z_bar;
        Vec pi = l[d].col(j).cwiseProduct(l[it.index()].col(j));
        p.eta(j, d) += (1 - Active(m.g_u, m.l_u, g_z_bar, l_z_bar, p.eta, d,
                     it.index()))*(m.l_u.col(j).dot(pi))*p.eta(j, it.index());
      }
    }
    NormalizeCol(d, &(p.eta));

    //omega
    p.omega(1, d) = m.gamma[1];
    for (size_t n = 0; n < doc.ULen(); n++) {
      p.omega(1, d) += p.delta[d][n]*doc.Count(n);
    }

    p.omega(0, d) = m.gamma[0];
    for (size_t n = 0; n < doc.ULen(); n++) {
      p.omega(0, d) += (1 - p.delta[d][n])*doc.Count(n);
    }

    //local theta
    for (int j = 0; j < m.LTopicNum1(); j++) {
      for (int k = 0; k < m.LTopicNum2(); k++) {
        p.l_theta[d](k, j) = p.eta(j, d) * m.l_alpha[j];
      }
    }
    for (int j = 0; j < m.LTopicNum1(); j++) {
      for (int k = 0; k < m.LTopicNum2(); k++) {
        for (size_t n = 0; n < doc.ULen(); n++) {
          p.l_theta[d](k, j) += doc.Count(n)* p.delta[d][n]*p.eta(j, d)*
                             p.l_z[d][j](k, n);
        }
        p.l_theta[d](k, j) += 1 - p.eta(j, d);
      }
    }

    //global theta
    p.g_theta.setConstant(m.g_alpha);
    for (size_t n = 0; n < doc.ULen(); n++) {
      for (int k = 0; k < m.GTopicNum(); k++) {
        p.g_theta(k, d) += doc.Count(n) * p.g_z[d](k, n) * (1 - p.delta[d][n]);
      }
    }
 
    for (size_t n = 0; n < doc.ULen(); n++) {
      //variable delta
      double tmp = DiGamma(m.gamma[1]) - DiGamma(m.gamma[0]);
      for (int j = 0; j < m.LTopicNum1(); j++) {
        for (int k = 0; k < m.LTopicNum2(); k++) {
          tmp += p.eta(j, d)*p.l_z[d][j](k, n)*l_theta_ep(k,j); 
          tmp += p.eta(j, d)*p.l_z[d][j](k, n)*m.l_ln_w[j](k, doc.Word(n));
        }
      }
      for (int k = 0; k < m.GTopicNum(); k++) {
        tmp -= p.g_z[d](k, n) * g_theta_ep[k];
        tmp -= p.g_z[d](k, n) * m.g_ln_w(k, doc.Word(n));
      }

      double label = 0;
      for (SpMatInIt it(net_, d); it; ++it) {
        double l = 0;
        l -= m.g_u.dot(p.g_z[d].col(n).cwiseProduct(g_z_bar.col(it.index())));
        for (int j = 0; j < m.LTopicNum1(); j++) {
          Vec pi = p.l_z[d][j].col(n).cwiseProduct(l_z_bar[it.index()].col(j));
          l += m.l_u.col(j).dot(pi)*p.eta(j, d)*p.eta(j, it.index());
        }
        label += l * (1 - Active(m.g_u, m.l_u, g_z_bar, l_z_bar, p.eta, d,
                          it.index()));
      }
      tmp += label / doc.TLen();
      p.delta[d][n] = Sigmoid(tmp);

      //local z
      for (int j = 0; j < m.LTopicNum1(); j++) {
        Vec gradient(m.LTopicNum2());
        gradient.setZero();
        for (SpMatInIt it(net_, d); it; ++it) {
          gradient += (1 - Active(m.g_u, m.l_u, g_z_bar, l_z_bar, p.eta,
                       d, it.index()))*p.eta(j, d)*l_z_bar[it.index()].col(j);
        }
        gradient = gradient.cwiseProduct(m.l_u.col(j));
        gradient /= cor_.TLen(d); 
        for (int k = 0; k < m.LTopicNum2(); k++) {
          p.l_z[d][j](k, n) = p.delta[d][n]*p.eta(j, d)*(l_theta_ep(k, j) +
                           m.l_ln_w[j](k, doc.Word(n))) + gradient[k];
        }
        NormalizeCol(n, &(p.l_z[d][j]));
      }

      //global z
      Vec gradient(m.GTopicNum());
      gradient.setZero();
      for (SpMatInIt it(net_, d); it; ++it) {
        gradient += (1 - Active(m.g_u, m.l_u, g_z_bar, l_z_bar, p.eta, d,
                     it.index()))*g_z_bar.col(it.index());
      }
      gradient = gradient.cwiseProduct(m.g_u);
      gradient /= cor_.TLen(d); 

      for (int k = 0; k < m.GTopicNum(); k++) {
        p.g_z[d](k, n) = (1 - p.delta[d][n])*(g_theta_ep[k] +
                      m.g_ln_w(k, doc.Word(n))) + gradient[k];
      }
      NormalizeCol(n, &(p.g_z[d]));
    }
  }
  return Likelihood(cor_.docs[d], p, m);
}

//  p.g_z_bar.col(d) = Mean(var.g_z, var.delta);
//  Mean(var.l_z, delta, eta, &(var.l_z_bar[d]));
//these action should be done in post 
void VarMGRTM::MStep(MGRSSC &ss, MGRTM* m) {
  //local
  for (int j = 0; j < m->LTopicNum1(); j++) {
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
  LearningEta(ss.g_z_bar, ss.l_z_bar, &(m->g_u), &(m->l_u));
  LOG(INFO) << "learning eta";
}

void AddNegative(SpMatC &src, int times, VTriple* des) {
  for (int d = 0; d < src.cols(); d++) { 
    SInt observed;
    for (SpMatInIt it(src, d); it; ++it) {
      observed.insert(it.index());
    }
    int size = observed.size();
    for (int i = 0; i < size * times; ++i) {
      int k = Random(src.rows());
      if(observed.find(k) == observed.end()) {
        des->push_back(Triple(k, d, -1));
        observed.insert(k);
      }
    }
  }
  for (int d = 0; d < src.cols(); d++) {
    for (SpMatInIt it(src, d); it; ++it) {
      des->push_back(Triple(it.index(), d, 1));
    }
  }
}

void VarMGRTM::Load(StrC &net_path, StrC &cor_path, int times) {
  cor_.LoadData(cor_path);
  SpMat network;
  ReadData(net_path, &network);

  VTriple vec;
  AddNegative(network, times, &vec);
  VTriple train;
  VTriple test;
  ::SplitData(vec, 0.8, &train, &test);
  net_.resize(network.rows(), network.cols());
  net_.setFromTriplets(train.begin(), train.end());
  held_out_net_.resize(network.rows(), network.cols());
  held_out_net_.setFromTriplets(test.begin(), test.end());
}

void VarMGRTM::Init(ConvergedC &converged) {
  converged_ = converged;
}

void VarMGRTM::AddPi(VecC &pi, int &feature_index, feature_node* x_space,
                                                   int &dim_index) const {
  for (int i = 0; i < pi.size(); ++i) {
    x_space[feature_index].index = dim_index;
    x_space[feature_index].value = pi[i];
    feature_index++;
    dim_index++;
  }
}
 
void VarMGRTM::LibLinearSample(MatC &g_z_bar, VMatC &l_z_bar,
                               feature_node* &x_space, problem* prob) const {
  // load positive sample from net between documents.
  int feature_index = 0;
  int sample_index = 0; //index for train data
  for (size_t d = 0; d < cor_.Len(); d++) {
    for (SpMatInIt it(net_, d); it; ++it) {
      int dim_index = 1;
      Vec pi = g_z_bar.col(d).cwiseProduct(g_z_bar.col(it.index()));
      prob->y[sample_index] = it.value();
      prob->x[sample_index] = &x_space[feature_index];
      AddPi(pi, feature_index, x_space, dim_index);
 
      for (int j = 0; j < l_z_bar[0].cols(); ++j) {
        Vec pi = l_z_bar[d].col(j).cwiseProduct(l_z_bar[it.index()].col(j));
        AddPi(pi, feature_index, x_space, dim_index);
      }   
      x_space[feature_index++].index = -1; 
      sample_index++;
    }   
  }
}

//p->n sample number, p->l feature number
//topic num is alpha.size()
void VarMGRTM::LearningEta(MatC &g_z_bar, VMatC &l_z_bar, Vec* g_u,
                                                          Mat* l_u) const {
  int feature_num = g_z_bar.rows() + l_z_bar[0].cols() * l_z_bar[0].rows();
  int training_data_num = net_.nonZeros();
  LOG(INFO) << training_data_num << " " << feature_num;
  long long elements = training_data_num * feature_num + training_data_num;
  LOG(INFO) << elements;
	
  parameter param;
  param.solver_type = L2R_LR;
  param.C = 1;
  param.eps = 0.01; // see setting below
  param.p = 0.1;
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;
		
  problem prob;
  prob.l = training_data_num;
  prob.bias = -1;
  prob.y = (double*)malloc(sizeof(double) * prob.l);
  prob.x = (struct feature_node **)malloc(sizeof(struct feature_node*) * prob.l);
  prob.n = feature_num;
  LOG(INFO) << elements;
  feature_node *x_space = (feature_node*)malloc(sizeof(feature_node)*elements);

  LibLinearSample(g_z_bar, l_z_bar, x_space, &prob);
  model* solver = train(&prob, &param);

  for (int i = 0; i < g_u->size(); ++i) {
    (*g_u)[i] = solver->w[i];
  }
  for (int j = 0; j < l_u->cols(); ++j) {
    for (int i = 0; i < l_u->rows(); ++i) {
      (*l_u)(i, j) = solver->w[g_u->size() + j * l_u->rows() + i];
    }
  }
  LOG(INFO) << "a";
  
  free(solver->w);
  free(solver->label);
  free(solver);
  free(prob.y);
  free(prob.x);
  free(x_space);
}

double VarMGRTM::PredictAUC(SpMatC &test, MGRTMC &m, Mat &g_z_bar, MatC &eta,
                            VMatC &l_z_bar) const {
  VReal real;
  VReal pre;
  for (int d = 0; d < test.cols(); d++) {
    for (SpMatInIt it(test, d); it; ++it) {
      real.push_back(it.value());
      pre.push_back(Active(m.g_u, m.l_u, g_z_bar, l_z_bar, eta, d, it.index()));
    }
  }
  return AUC(real,pre);
}
} // namespace ml 
