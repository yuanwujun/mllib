// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com
#ifndef ML_CROC_H_
#define ML_CROC_H_
#include "base_head.h"
#include "eigen.h"
struct T {
  double p;
  double r;
};

inline void CheckData(const VReal &r) {
  for (size_t i = 0; i < r.size(); i++) {
    if (r[i] != 1 && r[i] != 2) {
      LOG(INFO) << "real value is not 1 or 2";
      assert(false);
    }
  }
}

inline bool Great(const T &lhs, const T &rhs) {
   return lhs.p > rhs.p;
}

inline void Sort(const EVec &pre, const EVec &real, std::vector<T>* res) {
  std::vector<T> vec(pre.size());
  for (int i = 0; i < pre.size(); i++) {
    T t;
    t.p = pre[i];
    t.r = real[i];
    vec.push_back(t);
  }
  std::sort(vec.begin(), vec.end(), Great);
}

inline void CROC(const EMat &pre, const EMat &real, VVInt* res) {
  for (int i = 0; i < pre.cols(); i++) {
    std::vector<T> vec;
    Sort(pre.col(i), real.col(i), &vec);
    for (size_t j = 0; j < vec.size(); j++) {
      res->at(i).push_back(vec[j].r);
    }
  }
}

inline void SpSort(const SpVec &pre, const SpVec &real, std::vector<T>* d) {
  for (SpVec::InnerIterator it(pre), it2(real); it; ++it, ++it2) {
    T t;
    t.p = it.value();
    t.r = it2.value();
    d->push_back(t);
  }
  std::sort(d->begin(), d->end(), Great);
}

inline void CROC(const SpMat &pre, const SpMat &real, VVInt* res) {
  for (int i = 1; i < pre.cols(); i++) {
    std::vector<T> vec;
    SpSort(pre.col(i), real.col(i), &vec);
    VInt tmp;
    for (size_t j = 1; j < vec.size(); j++) {
      tmp.push_back(vec[j].r + 1);
    }
    res->push_back(tmp);
  }
}

inline void ToTal(const std::vector<T> &vec, int* r, int* p) {
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i].r == 1.0) {
      (*r)++;
    } else {
      (*p)++;
    }
  }
}

struct Point {
  int x;
  int y;
};

inline Str Join(const std::vector<Point> &p) {
  VVInt v;
  for (size_t i = 0; i < p.size(); i++) {
    VInt vv; 
    vv.push_back(p[i].x);
    vv.push_back(p[i].y);
    v.push_back(vv);
  }
  return Join(v, " ", "\n");
}

inline void CreatePoint(const std::vector<T> &vec, std::vector<Point>* p) {
  int tp = 0;
  int fp = 0;
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i].r >= 1.0) {
      tp++;
    } else {
      fp++;
    }
    Point t;
    t.x = fp;
    t.y = tp;
    p->push_back(t);
  }
}

inline double AUC(const std::vector<Point> &p, double t_total, double p_total) {
  double area = 0;
  for (size_t i = 1; i < p.size(); i++) {
    area += (p[i].y + p[i-1].y)*(p[i].x - p[i-1].x) / (2.0*t_total*p_total);
  }
  return area;
}

inline double AUC(const VReal &real, const VReal &pre) {
  std::vector<T> vec;
  for (size_t i = 0; i < pre.size(); i++) {
    T t;
    t.p = pre[i];
    t.r = real[i];
    vec.push_back(t);
  }
  std::sort(vec.begin(), vec.end(), Great);
  int t_total = 0;
  int p_total = 0;
  ToTal(vec, &t_total, &p_total);
  std::vector<Point> point;
  CreatePoint(vec, &point);
  return AUC(point, t_total, p_total);
}

inline void Sort(const VReal &real, const VReal &pre, std::vector<T>* des) {
  for (size_t i = 0; i < pre.size(); i++) {
    T t;
    t.p = pre[i];
    t.r = real[i];
    des->push_back(t);
  }
  std::sort(des->begin(), des->end(), Great);
}

inline void PrecisionRecall(const VReal &real, const VReal &pre, VVReal* curve) {
  std::vector<T> vec;
  Sort(real, pre, &vec);
  int t_total = 0;
  int p_total = 0;
  ToTal(vec, &t_total, &p_total);

  double t = 0;
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i].r == 2) {
      t++;
    }
    curve->at(0).push_back(t/(i + 1));
    curve->at(1).push_back(t/t_total);
  }
}

inline double F1Score(const VReal &real, const VReal &pre) {
  CheckData(real);
  VVReal curve(2);
  PrecisionRecall(real, pre, &curve);
  VReal f1(real.size());
  for (size_t i = 0; i < real.size(); i++) {
    double t = curve[0][i] + curve[1][i];
    if (t <= 0) {
      f1[i] = 0;
    } else {
      f1[i] = 2 * curve[0][i] * curve[1][i] / (curve[0][i] + curve[1][i]);
    }
  }
  VReal::iterator it = std::max_element(f1.begin(), f1.end());
  return *it;
}

inline double DecisionPro(const VReal &real, const VReal &pre) {
  std::vector<T> vec;
  Sort(real, pre, &vec);
  int t_total = 0;
  int p_total = 0;
  ToTal(vec, &t_total, &p_total);
  double t = 0;
  VVReal curve(2);
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i].r == 2) {
      t++;
    }
    curve.at(0).push_back(t/(i + 1));
    curve.at(1).push_back(t/t_total);
  }

  std::vector<T> f1;
  for (size_t i = 0; i < real.size(); i++) {
    T t;
    t.p = 2 * curve[0][i] * curve[1][i] / (curve[0][i] + curve[1][i]);
    t.r = i;
    f1.push_back(t);
  }
  std::sort(f1.begin(), f1.end(), Great);
  return vec[std::max_element(f1.begin(), f1.end(), Great)->r].p;
}
#endif // ML_CROC_H_
