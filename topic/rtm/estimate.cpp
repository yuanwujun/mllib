// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base_head.h"
#include "rtm.h"
#include "rtm_var_em.h"

using namespace ml;

DEFINE_string(cor_path, "rtm_network", "");
DEFINE_string(net_path, "rtm_corpus", "");
DEFINE_int32(topic_num, 10, "");
DEFINE_double(alpha, 0.01, "");

void App() 
{
  long t1;
  (void) time(&t1);
  seedMT(t1);
  float em_converged = 1e-4;
  int em_max_iter = 40;
  int em_estimate_alpha = 1; //1 indicate estimate alpha and 0 use given value
  int var_max_iter = 50;
  double var_converged = 1e-6;
  double initial_alpha = 0.1;
  VarRTM var;

  var.Init(em_converged, em_max_iter, em_estimate_alpha, var_max_iter,
                         var_converged, initial_alpha, FLAGS_topic_num);
  var.Load(FLAGS_net_path, FLAGS_cor_path);
  
  SpMat test;
  ReadData(FLAGS_net_path, &test);
  RTM rtm(FLAGS_topic_num, FLAGS_alpha);
  var.RunEM(test, &rtm);
}

int main(int argc, char* argv[]) 
{
  App();
  return 0; 
}
