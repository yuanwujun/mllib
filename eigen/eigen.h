// Copyright 2014 lijiankou. All Rights Reserved.
// author: lijk_start@163.com (jiankou li)
#ifndef ML_EIGEN_H_
#define ML_EIGEN_H_
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include "base_head.h"

typedef Eigen::SparseVector<double> SpVec;
typedef const SpVec SpVecC;
typedef SpVec::InnerIterator SpVecInIt;

typedef Eigen::SparseMatrix<double> SpMat;
typedef const SpMat SpMatC;
typedef SpMat::InnerIterator SpMatInIt;

typedef Eigen::MatrixXd EMat;
typedef Eigen::VectorXd EVec;
typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;
typedef std::vector<EMat> VMat;
typedef std::vector<VMat> VVMat;

typedef const Vec VecC;
typedef const Mat MatC;
typedef const VMat VMatC;
typedef const VVMat VVMatC;

typedef Eigen::Triplet<double> Triple;
typedef std::vector<Eigen::Triplet<double> > TripleVec;
typedef std::vector<Eigen::Triplet<double> > VTriple;

void ReadData(const Str &path, TripleVec* vec);
std::pair<int, int> Max(const TripleVec &vec);
std::pair<int, int> ReadData(const Str &path, SpMat* mat);
std::pair<int, int> ReadData(const Str &path, SpMat* train, SpMat* test);

void Sample(EVec *h);
void Sample(EMat *h);
void Sample(const EVec &src, EVec *des);
void NormalRandom(EMat *mat);
void NormalRandom(EVec *vec);
void SplitData(const TripleVec &vec, double p, TripleVec* train, TripleVec* test);

//m should be set size before, and must not muldify its size in ToEMat
void ToEMat(const VVReal &v_real, EMat* m);
void ToEVec(const VReal &v, EVec* m);
 
Str EVecToStr(const EVec &m);
Str EMatToStr(const EMat &m);
void StrToVReal(const Str &str, VReal* m);
void StrToVVReal(const Str &str, VVReal* m);

Str Join(const EMat &data);

//resize adj before
void CreateAdj(const SpMat &u_m, EMat* adj);

inline double InnerProd(const SpVec &vec, const EVec &v) {
  double sum = 0;
  for (SpVec::InnerIterator it(vec); it; ++it) {
    sum += it.value() * v[it.index()]; 
  }
  return sum;
}
#endif // ML_EIGEN_H_
