#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  float idx[N];
  for(int j=0; j<N; j++) {
    idx[j] = j;
  }

  __m256 jvec = _mm256_load_ps(idx);
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 fxvec = _mm256_load_ps(fx);
  __m256 fyvec = _mm256_load_ps(fy);

  for(int i=0; i<N; i++) {
    /* orig */
    // for(int j=0; j<N; j++) {
    //   // if(i != j) {
    //   //   float rx = x[i] - x[j];
    //   //   float ry = y[i] - y[j];
    //   //   float r = std::sqrt(rx * rx + ry * ry);
    //   //   fx[i] -= rx * m[j] / (r * r * r);
    //   //   fy[i] -= ry * m[j] / (r * r * r);
    //   // }
    // }

    __m256 ivec = _mm256_set1_ps(i);
    __m256 xi = _mm256_set1_ps(x[i]);
    __m256 yi = _mm256_set1_ps(y[i]);

    __m256 mask = _mm256_cmp_ps(ivec, jvec, _CMP_NEQ_OQ);
    __m256 rx = _mm256_sub_ps(xi, xvec); // x[i] - x[j]
    __m256 ry = _mm256_sub_ps(yi, yvec); // y[i] - y[j]
    __m256 tmp = _mm256_add_ps(_mm256_mul_ps(rx, rx), _mm256_mul_ps(ry, ry)); // rx*rx + ry*ry

    /* if use rsqrt -> get slightly different from the 3rd decimal points */
    __m256 r = _mm256_blendv_ps(_mm256_set1_ps(0), _mm256_rsqrt_ps(tmp), mask);
    tmp = _mm256_mul_ps(r, r); // r^2
    tmp = _mm256_mul_ps(tmp, r); // r^3
    __m256 tmpx = _mm256_mul_ps(_mm256_mul_ps(rx, mvec), tmp); // rx * m[j] / (r * r * r);
    __m256 tmpy = _mm256_mul_ps(_mm256_mul_ps(ry, mvec), tmp); // ry * m[j] / (r * r * r);

    /* if use sqrt -> get exactly the same */
    // __m256 r = _mm256_sqrt_ps(tmp);
    // tmp = _mm256_mul_ps(r, r);
    // tmp = _mm256_mul_ps(tmp, r);
    // __m256 tmpx = _mm256_blendv_ps(_mm256_set1_ps(0), _mm256_div_ps(_mm256_mul_ps(rx, mvec), tmp), mask);
    // __m256 tmpy = _mm256_blendv_ps(_mm256_set1_ps(0), _mm256_div_ps(_mm256_mul_ps(ry, mvec), tmp), mask);

    /* intrinsic reduction */ 
    __m256 bvec = _mm256_permute2f128_ps(tmpx, tmpx, 1);
    bvec = _mm256_add_ps(bvec, tmpx);
    bvec = _mm256_hadd_ps(bvec, bvec);
    bvec = _mm256_hadd_ps(bvec, bvec);
    tmpx = _mm256_sub_ps(fxvec, bvec);

    bvec = _mm256_permute2f128_ps(tmpy, tmpy, 1);
    bvec = _mm256_add_ps(bvec, tmpy);
    bvec = _mm256_hadd_ps(bvec, bvec);
    bvec = _mm256_hadd_ps(bvec, bvec);
    tmpy = _mm256_sub_ps(fyvec, bvec);  

    /* update val: use new val only at index=i (when mask is false) */
    fxvec = _mm256_blendv_ps(tmpx, fxvec, mask);
    fyvec = _mm256_blendv_ps(tmpy, fyvec, mask);
    _mm256_store_ps(fx, fxvec);
    _mm256_store_ps(fy, fyvec);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
