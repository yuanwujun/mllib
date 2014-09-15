// Copyright 2014 lijiankou. All Rights Reserved.
// author: lijk_start@163.com (jiankou li)
#include "eigen.h"

#include "util.h"

using std::map;
using std::pair;

void Sample(EVec *h) {
  for (int i = 0; i < h->size(); ++i) {
    (*h)[i] = Sample1((*h)[i]);
  }
}

void Sample(EMat *h) {
  for (int i = 0; i < h->cols(); ++i) {
    for (int j = 0; j < h->rows(); ++j) {
      h->col(i)[j] = Sample1(h->col(i)[j]);
    }
  }
}

void Sample(const EVec &src, EVec *des) {
  for (int i = 0; i < src.size(); ++i) {
    (*des)[i] = Sample1(src[i]);
  }
}

void ReadData(const Str &path, TripleVec* vec) {
  if (!IsFile(path)) {
    LOG(INFO) << path << " not exist"; 
    assert(false);
  }
  FILE *fin = fopen(path.c_str(), "r");
  int u;
  int v;
  float r;
  while(fscanf(fin, "%d %d %f", &u, &v, &r) > 0) {
    vec->push_back(Triple(v, u, r));
  }
}

std::pair<int, int> Max(const TripleVec &vec) {
  int col = 0;
  int row = 0;
  for (size_t i = 0; i < vec.size(); i++) {
    col = col > vec[i].col() ? col : vec[i].col();
    row = row > vec[i].row() ? row : vec[i].row();
  }
  return std::make_pair(row + 1, col + 1);
}

std::pair<int, int> ReadData(const Str &path, SpMat *mat) {
  TripleVec vec;
  ReadData(path, &vec);
  std::pair<int, int> p;
  if (mat->cols() == 0) {
    p = Max(vec);
    mat->resize(p.first, p.second);
  }
  mat->setFromTriplets(vec.begin(), vec.end());
  return p;
}

void NormalRandom(EVec* des) {
  for (int i = 0; i < des->size(); i++) {
    (*des)[i] = ml::NormalSample() / 100;
  }
}

void NormalRandom(EMat *mat) {
  for (int i = 0; i < mat->rows(); i++) {
    for (int j = 0; j < mat->cols(); j++) {
      (*mat)(i, j) = ml::NormalSample() / 100;
    }
  }
}

void SplitData(const TripleVec &vec, double p, TripleVec* train,
                                               TripleVec* test) {
  for (size_t i = 0; i < vec.size(); i++) {
    if (Random1() < p) {
      train->push_back(vec[i]);
    } else {
      test->push_back(vec[i]);
    }
  }
}

void Extend(const TripleVec &src, TripleVec* des) {
  for (size_t i = 0; i < src.size(); i++) {
    des->push_back(src[i]);
  }
}

void FilterData(const TripleVec &src,  size_t item_num, TripleVec* des) {
  map<int, TripleVec> dic;
  for (size_t i = 0; i < src.size(); i++) {
    dic[src[i].col()].push_back(src[i]);
  }
  int c = 0;
  for (map<int, TripleVec>::iterator it = dic.begin(); it != dic.end(); ++it) {
    if (it->second.size() > item_num) {
      c++;
      Extend(it->second, des);
    }
  }
}

pair<int, int> ReadData(const Str &path, SpMat* train, SpMat* test) {
  if (!IsFile(path)) {
    LOG(INFO) << path << " not exist"; 
    assert(false);
  }
  TripleVec vec;
  ReadData(path, &vec);
  double pro = 0.8;
  // int num = 20;
  // TripleVec filte;
  // FilterData(vec,  num, &filte);
  TripleVec train_vec;
  TripleVec test_vec;
  SplitData(vec, pro, &train_vec, &test_vec);
  pair<int, int> p;
  if (train->cols() == 0) {
    p = Max(vec);
    train->resize(p.first, p.second);
    test->resize(p.first, p.second);
  }
  test->setFromTriplets(test_vec.begin(), test_vec.end());
  train->setFromTriplets(train_vec.begin(), train_vec.end());
  LOG(INFO) << p.first << " " << p.second;
  return p;
}

void StrToVReal(const Str &str, VReal* m) {
  VStr v;
  SplitStr(str, " ", &v);
  m->reserve(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    if (!TrimStr(v[i]).empty()) {
      m->push_back(StrToReal(v[i]));
    }
  }
}

void StrToVVReal(const Str &str, VVReal* m) {
  VStr v;
  SplitStr(str, "\n", &v);
  m->reserve(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    if (!TrimStr(v[i]).empty()) {
      VReal v2;
      StrToVReal(v[i], &v2);
      m->push_back(v2);
    }
  }
}

void ToEMat(const VVReal &v_real, EMat* m) {
  for (int i = 0; i < m->rows(); i++) {
    for (int j = 0; j < m->cols(); j++) {
      (*m)(i, j) = v_real[i][j];
    }
  }
}

void ToEVec(const VReal &v, EVec* m) {
  for (int i = 0; i < m->size(); i++) {
    (*m)[i] = v[i];
  }
}

Str EVecToStr(const EVec &m) {
  VReal data(m.size());
  for (int i = 0; i < m.size(); i++) {
    data[i] = m[i];
  }
  return Join(data, " ");
}

Str EMatToStr(const EMat &m) {
  VVReal data(m.rows());
  for (int i = 0; i < m.rows(); i++) {
    VReal d2(m.cols());
    for (int j = 0; j < m.cols(); j++) {
      d2[j] = m(i, j);
    }
    data[i].swap(d2);
  }
  return Join(data, " ", "\n");
}

Str Join(const EMat &data) {
  VVReal d;
  for (int i = 0; i < data.rows(); i++) {
    VReal tmp;
    for (int j = 0; j < data.cols(); j++) {
      tmp.push_back(data(i, j)); 
    }
    d.push_back(tmp);
  }
  return Join(d, " ", "\n");
}

void CreateAdj(const SpMat &u_m, EMat* adj) {
  adj->setZero();
  for (int i = 0; i < u_m.cols(); i++) {
    for (SpMat::InnerIterator it(u_m, i); it; ++it) {
      (*adj)(it.index(), i) = it.value();
    }
  }
}
