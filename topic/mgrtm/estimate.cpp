// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base_head.h"
#include "converged.h"
#include "var_mgrtm.h"

using namespace ml;

DEFINE_string(cor_path, "./data/rtm_corpus", "");
DEFINE_string(net_path, "./data/rtm_network", "");
DEFINE_int32(topic_num, 10, "");
DEFINE_int32(neg_times, 10, "");
DEFINE_double(alpha, 0.01, "");

void MGRTMApp() {
  ml::Converged converged;
  converged.em_converged_ = 1e-4;
  converged.em_max_iter_ = 100;
  converged.var_converged_ = 1e-4;
  converged.var_max_iter_ = 10;

  VarMGRTM var;
  var.Init(converged);
  var.Load(FLAGS_net_path, FLAGS_cor_path, FLAGS_neg_times);

  Str path(FLAGS_cor_path);
  Corpus cor;
  cor.LoadData(path);

  LOG(INFO) << "b"; 
  MGRTM m;
  m.Init(2, 5, 5, cor.TermNum(), 1, 0.01, 0.01);
  var.RunEM(&m);
}

int main(int argc, char* argv[])  {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  //MGCTMApp();
  MGRTMApp();
  //RTMApp();
  return 0; 
}
