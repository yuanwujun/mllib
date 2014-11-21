// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base_head.h"
#include "btm_var_em.h"

using namespace ml;

DEFINE_string(cor_train, "/data0/data/btm/biterm", "");
DEFINE_string(cor_test, "/data0/data/btm/biterm", "");
DEFINE_double(beta, 0.01, "");
DEFINE_int32(em_iterate,30,"");
DEFINE_int32(var_iterate,30,"");
DEFINE_int32(topic_num, 10, "");

void LdaApp() {
  long t1;
  (void) time(&t1);
  seedMT(t1);

  VarBTM btm;
  btm.Init(FLAGS_em_iterate,
  FLAGS_var_iterate,FLAGS_topic_num,FLAGS_beta);
  btm.Load(FLAGS_cor_train);
  btm.RunEM();
}

int main(int argc, char* argv[]) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  LdaApp();

  return 1; 
}
