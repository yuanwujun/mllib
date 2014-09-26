#include "array_numeric_operator.h"

double double_array_sum(double* arr, int len) {
  double sum = 0.0;
  for(int i=0; i<len; ++i) {
    sum += arr[i];
  }
  return sum;
}
