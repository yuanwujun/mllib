// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base_head.h"
#include "converged.h"
#include "mgctm.h"
#include "var_mgctm.h"

using namespace ml;

DEFINE_string(cor_train, "/data0/data/comment/655/lda_data_2000", "");
DEFINE_string(cor_test, "/data0/data/comment/655/lda_data_2000", "");
DEFINE_int32(local_topic_num, 10, "");
DEFINE_int32(group, 10, "");
DEFINE_int32(global_topic_num, 10, "");
DEFINE_double(gamma, 1, "");
DEFINE_double(local_alpha, 0.01, "");
DEFINE_double(global_alpha, 0.01, "");
DEFINE_int32(em_iterate, 100, "");
DEFINE_int32(var_iterate, 100, "");

void MGCTMApp() {
  ml::Converged converged;
  converged.em_converged_ = 1e-4;
  converged.em_max_iter_ = FLAGS_em_iterate;
  converged.var_converged_ = 1e-4;
  converged.var_max_iter_ = FLAGS_var_iterate;

  Corpus train;
  Corpus test;
  train.LoadData(FLAGS_cor_train);
  test.LoadData(FLAGS_cor_test);

  VarMGCTM var;
  var.Init(converged);
  var.Load(FLAGS_cor_train);

  MGCTM m;
  m.Init(FLAGS_local_topic_num, FLAGS_group, FLAGS_global_topic_num, 
         train.TermNum(), FLAGS_gamma, FLAGS_local_alpha, FLAGS_global_alpha);

  var.RunEM(test,&m);
}

int main(int argc, char* argv[])  {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  MGCTMApp();
  return 0; 
}
