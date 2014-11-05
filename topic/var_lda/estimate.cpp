// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base_head.h"
#include "lda.h"
#include "lda_var_em.h"

using namespace ml;

DEFINE_string(cor_train, "/data0/data/comment/655/lda_data_2000", "");
DEFINE_string(cor_test, "/data0/data/comment/655/lda_data_2000", "");
DEFINE_double(alpha, 0.01, "");
DEFINE_int32(em_iterate,30,"");
DEFINE_int32(var_iterate,30,"");
DEFINE_int32(topic_num, 10, "");

void LdaApp() {
  long t1;
  (void) time(&t1);
  seedMT(t1);

  float em_converged = 1e-4;
  int em_max_iter = FLAGS_em_iterate;
  int em_estimate_alpha = 1; //1 indicate estimate alpha and 0 use given value
  int var_max_iter = FLAGS_var_iterate;
  double var_converged = 1e-6;
  double initial_alpha = FLAGS_alpha;
  int topic = FLAGS_topic_num;

  Corpus train;
  Corpus test;
  train.LoadData(FLAGS_cor_train);
  test.LoadData(FLAGS_cor_test);
  LOG(INFO) << train.Len()<< " " << test.Len();

  LdaModel m;
  LDA lda;
  lda.Init(em_converged, em_max_iter, em_estimate_alpha, var_max_iter,
                         var_converged, initial_alpha, topic);
  Str type = "seeded";
  lda.RunEM(type, train, test, &m);

  VVReal gamma;
  VVVReal phi;
  lda.Infer(test, m, &gamma, &phi);
  WriteStrToFile(Join(gamma, " ", "\n"), "./model/gamma");
  WriteStrToFile(Join(m.log_prob_w, topic, train.num_terms), "./model/beta");
  WriteStrToFile(Join(phi, " ", "\n", "\n\n"), "./model/phi");
}

int main(int argc, char* argv[]) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  LdaApp();

  return 1; 
}
