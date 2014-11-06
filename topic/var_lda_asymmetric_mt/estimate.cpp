// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base_head.h"
#include "lda.h"
#include "lda_var_em.h"

using namespace ml;

DEFINE_string(cor_train, "/data0/data/comment/mobilePhone/LDATrainData", "");
DEFINE_string(cor_test, "/data0/data/comment/mobilePhone/LDATestData", "");
DEFINE_double(alpha, 0.01, "");
DEFINE_int32(em_iterate,30,"");
DEFINE_int32(var_iterate,30,"");
DEFINE_int32(topic_num, 10, "");

void LdaApp() {
  long t1;
  (void) time(&t1);
  seedMT(t1);

  int em_max_iter = FLAGS_em_iterate;
  float em_converged = 1e-4;
  int var_max_iter = FLAGS_var_iterate;
  double var_converged = 1e-6;
  int topics = FLAGS_topic_num;
  int em_estimate_alpha = 1; //1 indicate estimate alpha and 0 use given value

  Corpus train;
  Corpus test;
  train.LoadData(FLAGS_cor_train);
  test.LoadData(FLAGS_cor_test);
  LOG(INFO) << train.Len()<< " " << test.Len();
  
  double* alpha = new double[topics];
  for(int k=0; k<topics; ++k) {
    alpha[k] = FLAGS_alpha;
  }
  LdaModel m;
  m.InitModel(topics,train.num_terms,alpha);
  
  LDA lda;
  lda.Init(em_converged, em_max_iter, em_estimate_alpha, var_max_iter,var_converged);  
  Str type = "seeded";  
  lda.RunEM(type, train, test, &m);

  VVReal gamma;
  VVVReal phi;
  lda.Infer(test, m, &gamma, &phi);
  WriteStrToFile(Join(gamma, " ", "\n"), "./model/gamma");
  WriteStrToFile(Join(m.log_prob_w, topics, train.num_terms), "./model/beta");
  WriteStrToFile(Join(phi, " ", "\n", "\n\n"), "./model/phi");

  delete [] alpha;
}

int main(int argc, char* argv[]) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  LdaApp();

  return 1; 
}
