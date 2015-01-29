// Copyright 2014 yuanwujun. All Rights Reserved.
// Author: real.yuanwj@gmail.com

#include <algorithm>
#include "ftrl.h"

void ftrl_proximal::init(float alpha, float beta, float lambda1, float lambda2, long feature_dim)
{
    alpha_ = alpha;
    beta_ = beta;
    lambda1_ = lambda1;
    lambda2_ = lambda2;
    feature_dim_ = feature_dim;

    n_.resize(feature_dim_);
    z_.resize(feature_dim_);
    w_.resize(feature_dim_);

    n_.setZero();
    z_.setZero();
    w_.setZero();
}

float ftrl_proximal::predict(SparseVector<float> x)
{
    double wTx = 0.0;

    for (SparseVector<float>::InnerIterator it(x); it; ++it)
    {
        int i = it.index();
        int sign = 0;
        if(z_(i) < 0)
            sign =-1;
        else 
            sign = 1;

        if(z_(i) * sign <= lambda1_)
            w_(i) = 0.0;
        else 
            w_(i) = (sign * lambda1_ - z_(i)) / ((beta_ + sqrt(n_(i))) / alpha_ + lambda2_);

        wTx += w_(i) * it.value();
    }

    return 1.0 / (1.0 + exp(-max(min(wTx, 35.0), -35.0)));
}

void ftrl_proximal::update(SparseVector<float> x, float p, float y)
{
    for (SparseVector<float>::InnerIterator it(x); it; ++it)
    {
        int i = it.index();
        float x_i = it.value();
        float g = (p - y) * x_i;
        float sigma = (sqrt(n_(i) + g * g) - sqrt(n_(i))) / alpha_;
        z_(i) += g - sigma * w_(i);
        n_(i) += g * g;
    }
}

float ftrl_proximal::logloss(double p, double y)
{
    float loss = 0.0;

    p = max(min(p, 1.0 - 10e-15), 10e-15);
    if (1.0 == y) 
        loss = -log(p);
    else 
        loss = -log(1.0 - p);

    return  loss;
}
