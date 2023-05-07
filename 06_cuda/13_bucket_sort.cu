#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_init(int *bucket) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  bucket[i] = 0;
}


__global__ void bucket_update(int *bucket, int *key) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int key_shared[];
  __syncthreads();
  key_shared[threadIdx.x] = key[i];
  __syncthreads();
  atomicAdd(&bucket[key_shared[i]], 1);
}


__global__ void key_update(int *bucket, int *key, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  /* create and calculate offset */
  extern __shared__ int s[];
  int *integerData = s;                    
  int *offset = (int*)&integerData[range];
  int *offset_copy = (int*)&integerData[range];

  __syncthreads();
  offset[i] = 0;
  offset_copy[i] = 0;
  __syncthreads();

  for (int j=1; j<range; j<<=1) {
    offset_copy[i] = offset[i];
    __syncthreads();
    if (i >= j)
      offset[i] += offset_copy[i-j] + bucket[i-j];
    __syncthreads();
  }
  __syncthreads();

  /* update key based on offset and bucket */
  int j = offset[i];
  for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
  }
}


int main() {
  int n = 50;
  int range = 5;
  // std::vector<int> key(n);
  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  /* 
  std::vector<int> bucket(range); 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
  */

  int *bucket;
  cudaMallocManaged(&bucket, range*sizeof(int));
  bucket_init<<<1,range>>>(bucket);
  cudaDeviceSynchronize();

  bucket_update<<<1,n,n*sizeof(int)>>>(bucket, key);
  cudaDeviceSynchronize();

  key_update<<<1,range, 2*range*sizeof(int)>>>(bucket, key, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(bucket);
  cudaFree(key);
}
