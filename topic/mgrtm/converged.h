// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef TOPIC_CONVERGED_H_
#define TOPIC_CONVERGED_H_

namespace ml {
struct Converged {
  int em_max_iter_;
  double em_converged_;
  int var_max_iter_;
  double var_converged_;
};
typedef const Converged ConvergedC;
}  // namespace ml
#endif // TOPIC_CONVERGED_H_
