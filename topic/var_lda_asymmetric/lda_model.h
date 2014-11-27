// Copyright 2014 yuanwujun. All Rights Reserved.
// Author: real.yuanwj@gmail.com (WuJun Yuan)
#ifndef ML_LDA_LDA_MODEL_H_
#define ML_LDA_LDA_MODEL_H_
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "base_head.h"
#include "cokus.h"
#include "lda.h"

namespace ml {
double OptAlpha(const LdaSuffStats &ss, LdaModel* m);
void LdaMLE(int estimate_alpha, const LdaSuffStats &ss, LdaModel* m);
}  // namespace ml
#endif  // ML_LDA_LDA_MODEL_H_
