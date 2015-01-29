// Copyright 2014 yuanwujun. All Rights Reserved.
// Author: real.yuanwj@gmail.com

#ifndef __FTRL_H__
#define __FTRL_H__

#include "Eigen/Dense"
#include "Eigen/Sparse"

using namespace Eigen;
using namespace std;

class ftrl_proximal {
public:
    void init(float alpha, float beta, float lambda1, float lambda2, long feature_dim);
    float predict(SparseVector<float> x);
    void update(SparseVector<float> x, float p, float y);
    float logloss(double p,double y);
  
private:
    float alpha_;
    float beta_;
    float lambda1_;
    float lambda2_;
    long feature_dim_;

    VectorXf n_;
    VectorXf z_;
    VectorXf w_;
};

#endif //__FTRL_H__
