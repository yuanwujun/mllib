// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base_head.h"
#include "converged.h"
#include "mgctm.h"
#include "var_mgctm.h"

using namespace ml;

DEFINE_string(cor_path, "./data/rtm_corpus", "");
DEFINE_string(net_path, "./data/rtm_network", "");
DEFINE_int32(topic_num, 10, "");
DEFINE_double(alpha, 0.01, "");

void MGCTMApp() {
  ml::Converged converged;
  converged.em_converged_ = 1e-4;
  converged.em_max_iter_ = 100;
  converged.var_converged_ = 1e-4;
  converged.var_max_iter_ = 10;

  VarMGCTM var;
  var.Init(converged);

  Str path(FLAGS_cor_path);
  Corpus cor;
  cor.LoadData(path);

  var.Load(path);
  LOG(INFO) << path;

  MGCTM m;
  m.Init(5, 5, 10, cor.TermNum(), 0.1, 0.1, 0.1);
  LOG(INFO) << "init over";

  var.RunEM(&m);
}

int main(int argc, char* argv[])  {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  MGCTMApp();
  return 0; 
}
