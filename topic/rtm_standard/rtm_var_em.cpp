// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "rtm_var_em.h"

#include <omp.h>
#include "base_head.h"
#include "eigen.h"
#include "eigen_util.h"
#include "rtm.h"

#include "linear.h"

#include "roc.h"

namespace ml {
void ExpectTheta(VecC &ga, Vec* des) {
  double digsum = DiGamma(ga.sum());
  des->resize(ga.size());
  for (int k = 0; k < ga.size(); k++) {
    (*des)[k] = DiGamma(ga[k]) - digsum;
  }
}

/*****
Infer and compute suffstats, the motivation of infer is computing suffstats
phi: topic * doc_len
update z_bar
*****/
void VarRTM::EStep(CorpusC &cor, RTMC &m, RTMSuffStats* ss) const {
  RTMVar var;
  Infer(cor, m, &var);
  ss->z_bar.resize(m.TopicNum(), cor.Len());
  for (size_t d = 0; d < cor.Len(); d++) {
    for (size_t n = 0; n < cor.ULen(d); n++) {
      for (int k = 0; k < m.TopicNum(); k++) {
        ss->topic(k, cor.Word(d, n)) += cor.Count(d, n) * var.phi[d](k, n);
        ss->topic_sum[k] += cor.Count(d, n) * var.phi[d](k, n);
      }
    }
    ss->z_bar.col(d) = var.z_bar.col(d);
  }
}

void VarRTM::RunEM(RTM* m) {
  m->Init(cor.TermNum());
  RTMSuffStats ss;
  ss.InitSS(m->TopicNum(), m->TermNum());
  MStep(ss, m);
  m->alpha = initial_alpha_;
  for (int i = 0; i < em_max_iter_; i++) {
    ss.SetZero(m->TopicNum(), m->TermNum());
    EStep(cor, *m, &ss);
    MStep(ss, m);
    Vec alpha(m->TopicNum());
    LearningEta(alpha, ss.z_bar, &(m->eta));
    LOG(INFO) << i << " AUC:" << PredictAUC(*m, ss.z_bar);
  }
}

/****
phi: dimensions is doc_id, matrix is topic * doc_size 
z_bar: topic*doc_id
****/
void VarRTM::Infer(CorpusC &cor, RTMC &m, RTMVar* var) const {
  var->Init(cor, m);
  for(int it = 1; it < var_max_iter_; ++it) {
    for (size_t d = 0; d < cor.Len(); d++) {
      for(int it2 = 1; it2 < doc_var_max_iter_; ++it2) {
        Vec vec(m.TopicNum());
        vec.setZero();
        for (SpMatInIt it(net, d); it; ++it) {
          Vec pi = var->z_bar.col(d).cwiseProduct(var->z_bar.col(it.index()));
          vec += (1 - Sigmoid(m.eta.dot(pi)))*var->z_bar.col(it.index());
        }
        Vec gradient = vec.cwiseProduct(m.eta);
        gradient /= cor.TLen(d); //every word is considered to be different
        Vec expect_theta;
        ExpectTheta(var->gamma.col(d), &expect_theta);
        for (size_t n = 0; n < cor.ULen(d); n++) {
          for (int k = 0; k < m.TopicNum(); k++) {
            (var->phi)[d](k, n) = expect_theta[k] + 
                                m.ln_w(k, cor.Word(d, n)) + gradient[k];
          }
          double ln_phi_sum = LogSum((var->phi)[d].col(n));
          for (int k = 0; k < m.TopicNum(); k++) { //normalize phi
            (var->phi)[d](k, n) = exp((var->phi)[d](k, n) - ln_phi_sum);
          }
        }
        var->gamma.setConstant(m.alpha);
        for (size_t n = 0; n < cor.ULen(d); n++) {
          for (int k = 0; k < m.TopicNum(); k++) {
            var->gamma(k, d) += cor.Count(d, n) * (var->phi)[d](k, n);
          }
        }
        var->z_bar.col(d) = ZBar(cor.docs[d], var->phi[d]);
      }
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

double VarRTM::PredictAUC(RTMC &m, Mat &z_bar) {
  VReal real,pre;
  for (int d = 0; d < held_out_net_.cols(); d++) {
    for (SpMatInIt it(held_out_net_, d); it; ++it) {
      double label = it.value();
      Vec pi = z_bar.col(d).cwiseProduct(z_bar.col(it.index()));
      double prob = Sigmoid(pi.dot(m.eta));
      real.push_back(label);
      pre.push_back(prob);
    }
  }
  return AUC(real,pre);
}

void VarRTM::Load(StrC &net_path, StrC &cor_path) {
  ReadData(net_path, &net, &held_out_net_);
  cor.LoadData(cor_path);

  SpMat all_network;
  ReadData(net_path, &all_network);
  TripleVec vec;
  for (int d = 0;d < all_network.cols(); d++) { 
    SInt observed;
    for (SpMatInIt it(all_network, d); it; ++it) {
      int row = it.row();
      observed.insert(row);
    }
    int observed_size = observed.size();
    int nonobserved_size = all_network.rows() - observed_size;
    for (int i = 0;i < observed_size * (nonobserved_size / 11000); ++i) {
      int k = Random(all_network.rows());
      if(observed.find(k) == observed.end()) {
        vec.push_back(Triple(k, d, -1));
        observed.insert(k);
      }
    }
  }
  for (int d = 0;d < held_out_net_.cols(); d++) {
    for (SpMatInIt it(held_out_net_, d); it; ++it) {
      vec.push_back(Triple(it.row(), d, 1));
    }
  }
  held_out_net_.setFromTriplets(vec.begin(), vec.end());

  LOG(INFO) << cor_path;
  LOG(INFO) << cor.Len();
  LOG(INFO) << net.size();
  LOG(INFO) << held_out_net_.size(); 
}

//p->n sample number, p->l feature number
//topic num is alpha.size()
void VarRTM::LearningEta(VecC &alpha, const Mat &z_bar, Vec *eta) const {
  int feature = alpha.size();
  int non_zero_num_in_net = net.nonZeros();
  int negative_sample_num = non_zero_num_in_net * rho_;
  int training_data_num = non_zero_num_in_net + negative_sample_num;
  long elements = training_data_num * feature + training_data_num;
  double negative_value_dim = 1.0 / (feature * feature);
	
  struct parameter param;
  param.solver_type = L2R_LR;
  param.C = 1;
  param.eps = 0.01; // see setting below
  param.p = 0.1;
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;
		
  struct problem prob;
  prob.l = training_data_num;
  prob.bias = -1;
  prob.y = (double*)malloc(sizeof(double) * prob.l);
  prob.x = (struct feature_node **)malloc(sizeof(struct feature_node *) * prob.l);
  prob.n = feature;
	
  struct feature_node *x_space;
  x_space = (feature_node *)malloc(sizeof(feature_node) * elements);
	
  // load positive sample from net between documents.
  int current_feature_node_index = 0;
  int current_l = 0;
  for (size_t d = 0; d < cor.Len(); d++) {
    for (SpMatInIt it(net, d); it; ++it) {
      Vec pi = z_bar.col(d).cwiseProduct(z_bar.col(it.index()));
			
      prob.y[current_l] = 1;
      prob.x[current_l] = &x_space[current_feature_node_index];
      for (int i = 0;i < feature;++i) {
        x_space[current_feature_node_index].index = i+1;
        x_space[current_feature_node_index].value = pi(i);
        current_feature_node_index ++;
      }
      x_space[current_feature_node_index++].index = -1;
      current_l ++;
    }
  }
  
  // construct negative sample from Dirichlet prior of the model.
  for(int neg_i = 0;neg_i < negative_sample_num;++neg_i) {
  	prob.y[current_l] = -1;
  	prob.x[current_l] = &x_space[current_feature_node_index];
  	for(int i = 0;i < feature;++i) {
  		x_space[current_feature_node_index].index = i+1;
		x_space[current_feature_node_index].value = negative_value_dim;
		current_feature_node_index ++;
  	}
    x_space[current_feature_node_index++].index = -1;
    current_l ++;
  }
  
  struct model* solver=train(&prob, &param);
  for (int i = 0;i < feature;++i) {
  	(*eta)[i] = solver->w[i];
  }
  
  free(solver->w);
  free(solver->label);
  free(solver);
  free(prob.y);
  free(prob.x);
  free(x_space);
}
} // namespace ml 
