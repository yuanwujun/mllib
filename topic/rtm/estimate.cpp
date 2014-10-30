// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base_head.h"
#include "rtm.h"
#include "rtm_var_em.h"

using namespace ml;

DEFINE_string(cor_path, "/data0/data/rtm/corpus", "");
DEFINE_string(net_path, "/data0/data/rtm/link", "");
DEFINE_int32(topic_num, 10, "");
DEFINE_double(alpha, 0.01, "");

int main(int argc, char* argv[])  {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
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

  int times = 10;

  var.Init(em_converged, em_max_iter, em_estimate_alpha, var_max_iter,
                         var_converged, initial_alpha, FLAGS_topic_num,1);
  LOG(INFO) << FLAGS_net_path;
  LOG(INFO) << FLAGS_cor_path;
  var.Load(FLAGS_net_path, FLAGS_cor_path, times);
  
  RTM rtm(FLAGS_topic_num, FLAGS_alpha);
  var.RunEM(&rtm);

  return 0; 
}
