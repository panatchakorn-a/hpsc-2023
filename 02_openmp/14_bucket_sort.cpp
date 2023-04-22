#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
  for (int i=0; i<n; i++)
    bucket[key[i]]++;
  std::vector<int> offset(range,0);
  std::vector<int> offset_copy(range,0);

// initialize offset with values from bucket
#pragma omp parallel for
  for (int i=1; i<range; i++)
    offset[i] = bucket[i-1];

// calculate offset by prefix sum
#pragma omp parallel
  for (int k=1; k<range; k<<=1) {
#pragma omp for
    for (int i=0; i<range; i++)
      offset_copy[i] = offset[i];
#pragma omp for
    for (int i=k; i<range-1; i++)
      offset[i+1] += offset_copy[i+1-k];
  }

// update sorted values based on values from bucket and offset
#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
