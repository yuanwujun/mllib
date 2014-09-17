// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "rtm_var_em.h"

#include <omp.h>
#include "base_head.h"
#include "eigen.h"
#include "eigen_util.h"
#include "rtm.h"

#include "linear.h"

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
  double likelihood = Infer(d, m, *z_bar, &gamma, &phi);
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
//double VarRTM::Infer(int d, RTMC &m, Vec* ga, Mat* phi, Mat* z_bar) const {
double VarRTM::Infer(int d, RTMC &m, MatC &z_bar, Vec* ga, Mat* phi) const {
  double likelihood_old = 0;
  double c = 1;
  Vec digamma;
  InitVar(d, m, &digamma, ga, phi);
  Vec vec(m.TopicNum());  //allociate one time
  for(int it = 1; (c > var_converged_) && (it < var_max_iter_); ++it) {
    vec.setZero();
    for (SpMatInIt it(net, d); it; ++it) {
      vec += z_bar.col(it.index());
    }

    /*
    for (SpMatInIt it(net, d); it; ++it) {
      Vec pi = z_bar.col(d).cwiseProduct(z_bar.col(it.index()));
      vec += (1 - Sigmoid(m.eta.dot(pi)))*pi;
    }
    */
    Vec gradient = vec.cwiseProduct(m.eta);
    gradient /= cor.TLen(d); //every word is considered to be different
    for (size_t n = 0; n < cor.ULen(d); n++) {
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
      }
    }
    for (int k = 0; k < m.TopicNum(); k++) {
      digamma[k] = DiGamma((*ga)[k]);
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
    Infer(i, m, z_bar, &gamma, &phi);
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
void VarRTM::LiblinearInputData(VecC &alpha, const Mat &z_bar, Vec *eta) const {
	int feature = alpha.size();
	int non_zero_num_in_net = net.nonZeros();
	int negative_sample_num = non_zero_num_in_net * rho;
	int training_data_num = non_zero_num_in_net + negative_sample_num;
	long elements = training_data_num * feature;
	double negative_value_dim = 1.0 / (feature * feature);
	
	struct parameter param;
	param.solver_type = L2R_LR;
	param.C = 0;
	param.eps = 0.01; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
		
	struct problem prob;
	prob.l = training_data_num;
	prob.bias = 0;
	prob.y = (double)malloc(sizeof(double) * prob.l);
	prob.x = (struct feature_node *)malloc(sizeof(struct feature_node *) * prob.l);
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
				x_space[current_feature_node_index].index = i;
				x_space[current_feature_node_index].value = pi(i);
				current_feature_node_index ++;
			}
			current_l ++;
    }
  }
  
  // construct negative sample from Dirichlet prior of the model.
  for(int neg_i = 0;neg_i < negative_sample_num;++neg_i) {
  	prob.y[current_l] = -1;
  	prob.x[current_l] = &x_space[current_feature_node_index];
  	for(int i = 0;i < feature;++i) {
  		x_space[current_feature_node_index].index = i;
			x_space[current_feature_node_index].value = negative_value_dim;
			current_feature_node_index ++;
  	}
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
	free(param->weight_label);
	free(param->weight);
}
} // namespace ml 
