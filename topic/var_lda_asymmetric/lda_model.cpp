#include "lda_model.h"

#include "array_numeric_operator.h"

#include "base_head.h"

namespace ml {
const int NUM_INIT = 1;
void LdaMLE(int estimate_alpha, const LdaSuffStats &ss, LdaModel* m) {
  for (int k = 0; k < m->num_topics; k++) {
    for (int w = 0; w < m->num_terms; w++) {
      if (ss.class_word[k][w] > 0) {
        m->log_prob_w[k][w] = log(ss.class_word[k][w]) - log(ss.class_total[k]);
      } else {
        m->log_prob_w[k][w] = -100;
      }
    }
  }
  if (estimate_alpha == 1) {
    OptAlpha(ss, m);
  }
}

double Alhood(const LdaSuffStats &ss, LdaModel* m) {
  double alpha_sum = double_array_sum(m->alpha,m->num_topics);
  double log_gamma_alpha_component = 0.0;
  double alpha_component_sufficient_product = 0.0;
  for(int k=0; k<m->num_topics; ++k) {
    log_gamma_alpha_component += lgamma(m->alpha[k]);
    alpha_component_sufficient_product += ( m->alpha[k] - 1 ) * ss.alpha_suffstats[k];
  }
  return (ss.num_docs * lgamma(alpha_sum) -  ss.num_docs * log_gamma_alpha_component + alpha_component_sufficient_product);
}

double DAlhood(const LdaSuffStats &ss, double* alpha,int topic, double* gradient) {
  double alpha_sum = double_array_sum(alpha,topic);
  for(int k=0; k<topic; ++k) {
    gradient[k] = ss.num_docs * (DiGamma(alpha_sum) - DiGamma(alpha[k])) + ss.alpha_suffstats[k];
  }
  return 0.0;
}

double D2Alhood(double* alpha, int d, int k, double* matrix_diag) {
  double alpha_sum = double_array_sum(alpha,k);
  for(int i=0; i<k; ++i) {
    matrix_diag[i] = d * TriGamma(alpha[i]) - TriGamma(alpha_sum);
  }
  return TriGamma(alpha_sum) * (-1);
}

double OptAlpha(const LdaSuffStats &ss, LdaModel* m) {
//  const double  NEWTON_THRESH = 1e-5;
  const int MAX_ALPHA_ITER = 1000;
  int iter = 0;
  double* gradient = new double[m->num_topics];
  double* matrix_diag = new double[m->num_topics];
  double* alpha = new double[m->num_topics];
  for(int k=0; k<m->num_topics; ++k) {
    alpha[k] = 100;
  }

  do {
    iter++;

    DAlhood(ss,alpha,m->num_topics,gradient);
    double non_diag = D2Alhood(alpha,ss.num_docs,m->num_topics,matrix_diag);
    
    double diag_sum = 0.0;
    double gradient_diag_ritio_sum = 0.0;
    for(int j=0; j<m->num_topics; ++j) {
      diag_sum += 1 / matrix_diag[j];
      gradient_diag_ritio_sum += gradient[j]/matrix_diag[j];
    }
    double c = gradient_diag_ritio_sum / ( 1 / non_diag + diag_sum);
    
    for(int k=0; k<m->num_topics; ++k) {
      m->alpha[k] = m->alpha[k] - (gradient[k] - c) / matrix_diag[k];
    }
  } while (/*(fabs(df) > NEWTON_THRESH) && */(iter < MAX_ALPHA_ITER));

  return 0.0;
}
}  // namespace ml
