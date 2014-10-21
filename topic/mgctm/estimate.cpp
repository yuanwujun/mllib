// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base_head.h"
#include "converged.h"
#include "mgctm.h"
#include "var_mgctm.h"

using namespace ml;

DEFINE_string(cor_path, "/data0/data/order3/order3_8000_lda_data", "");
DEFINE_int32(local_topic_num, 10, "");
DEFINE_int32(group, 10, "");
DEFINE_int32(global_topic_num, 10, "");
DEFINE_double(gamma, 1, "");
DEFINE_double(local_alpha, 0.01, "");
DEFINE_double(global_alpha, 0.01, "");

void MGCTMApp() {
  ml::Converged converged;
  converged.em_converged_ = 1e-4;
  converged.em_max_iter_ = 100;
  converged.var_converged_ = 1e-4;
  converged.var_max_iter_ = 5;

  VarMGCTM var;
  var.Init(converged);

  Str path(FLAGS_cor_path);
  Corpus cor;
  cor.LoadData(path);
  LOG(INFO) <<cor.Len();

  var.Load(path);
  LOG(INFO) << path;

  MGCTM m;
  m.Init(FLAGS_local_topic_num, FLAGS_group, FLAGS_global_topic_num, cor.TermNum(), FLAGS_gamma, FLAGS_local_alpha, FLAGS_global_alpha);
  LOG(INFO) << "init over";
  
  Corpus train;
  Corpus test;
  double p = 0.8;
  SplitData(cor, p, &train, &test);
  Str type = "seeded";
              
  //Corpus test;
  //var.RunEM(test,&m);
   var.RunEM(test,&m);
}

int main(int argc, char* argv[])  {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  MGCTMApp();
  return 0; 
}
