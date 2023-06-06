#include <cstdlib>
#include <cstdio>
#include <vector>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int nx = 161; //41
const int ny = 161; //41
int nt = 10; //500
int nit = 50;
float dx = 2. / (nx - 1);
float dy = 2. / (ny - 1);
float dt = 0.01;
float rho = 1;
const float nu = 0.02;


__global__ void update_b(float *b, float *u, float *v, float rho, float dx, float dy, float dt){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i<=0 || i>=nx-1 || j<=0 || j>=ny-1) return;
    
    b[j*nx+i] = rho * (1 / dt *
                    ((u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx) + (v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy)) -
                    ((u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx))*((u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx)) - 
                    2 * ((u[(j+1)*nx+i] - u[(j-1)*nx+i]) / (2 * dy) *
                    (v[j*nx+i+1] - v[j*nx+i-1]) / (2 * dx)) - 
                    ((v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy))*((v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy)));
}


__global__ void init_pn(float *p, float *pn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    pn[j*nx+i] = p[j*nx+i];
}


__global__ void update_p(float *p, float *pn, float *b, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i<=0 || i>=nx-1 || j<=0 || j>=ny-1) return;

    /* update non-boundary values */
    p[j*nx+i] = (dy*dy * (pn[j*nx+i+1] + pn[j*nx+i-1]) +
            dx*dx * (pn[(j+1)*nx+i] + pn[(j-1)*nx+i]) -
            b[j*nx+i] * dx*dx * dy*dy)/(2 * (dx*dx + dy*dy));   
    __syncthreads();

    /* update boundary values */
    p[j*nx+nx-1] = p[j*nx+nx-2];
    p[j*nx] = p[j*nx+1];
    __syncthreads();
    p[i] = p[nx+i];
    p[(ny-1)*nx+i] = p[(ny-2)*nx+i];

}


__global__ void init_unvn(float *u, float *un, float *v, float *vn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    un[j*nx+i] = u[j*nx+i];
    vn[j*nx+i] = v[j*nx+i];
}


__global__ void update_uv(float *u, float *un, float *v, float *vn, float *p, float rho, float dx, float dy, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i<=0 || i>=nx-1 || j<=0 || j>=ny-1) return;

    /* update non-boundary values */
    u[j*nx+i] = un[j*nx+i] - un[j*nx+i] * dt / dx * (un[j*nx+i] - un[j*nx+i-1])
                    - un[j*nx+i] * dt / dy * (un[j*nx+i] - un[(j-1)*nx+i])
                    - dt / (2 * rho * dx) * (p[j*nx+i+1] - p[j*nx+i-1])
                    + nu * dt / dx*dx * (un[j*nx+i+1] - 2 * un[j*nx+i] + un[j*nx+i-1])
                    + nu * dt / dy*dy * (un[(j+1)*nx+i] - 2 * un[j*nx+i] + un[(j-1)*nx+i]);
    v[j*nx+i] = vn[j*nx+i] - vn[j*nx+i] * dt / dx * (vn[j*nx+i] - vn[j*nx+i-1])
                    - vn[j*nx+i] * dt / dy * (vn[j*nx+i] - vn[(j-1)*nx+i])
                    - dt / (2 * rho * dx) * (p[(j+1)*nx+i] - p[(j-1)*nx+i])
                    + nu * dt / dx*dx * (vn[j*nx+i+1] - 2 * vn[j*nx+i] + vn[j*nx+i-1])
                    + nu * dt / dy*dy * (vn[(j+1)*nx+i] - 2 * vn[j*nx+i] + vn[(j-1)*nx+i]);
    __syncthreads();

    /* update boundary values */
    u[j*nx] = 0;
    u[j*nx+nx-1] = 0;
    v[j*nx] = 0;
    v[j*nx+nx-1] = 0;
    __syncthreads();
    u[i] = 0;
    u[(ny-1)*nx+i] = 1;
    v[i] = 0;
    v[(ny-1)*nx+i] = 0;

}


long time_diff_sec(struct timeval st, struct timeval et)
{
    return (et.tv_sec-st.tv_sec)*1000000 + (et.tv_usec-st.tv_usec);
}



int main() {
    /* variables preparation */
    const int BS = 32; // num threads BS*BS per block
    dim3 grid = dim3((nx+BS-1)/BS, ((ny+BS-1)/BS), 1); // dim of num blocks per grid
    dim3 block = dim3(BS, BS, 1); // dim of num threads per block
    
    /* cuda variables */
    float *u, *v, *b, *p;
    float *un, *vn, *pn;
    cudaMallocManaged(&u, ny*nx*sizeof(float));
    cudaMallocManaged(&v, ny*nx*sizeof(float));
    cudaMallocManaged(&b, ny*nx*sizeof(float));
    cudaMallocManaged(&p, ny*nx*sizeof(float));
    cudaMallocManaged(&un, ny*nx*sizeof(float));
    cudaMallocManaged(&vn, ny*nx*sizeof(float));
    cudaMallocManaged(&pn, ny*nx*sizeof(float));

    struct timeval st1, st2, st3, et1, et2, et3;
    long t1, t2, t3;

    for (int n=0; n<nt; n++) {
        /* update b */
        gettimeofday(&st1, NULL);
        update_b<<<grid, block>>>(b, u, v, rho, dx, dy, dt);
        cudaDeviceSynchronize();
        gettimeofday(&et1, NULL);

        /* Poisson; update pressure p  */
        gettimeofday(&st2, NULL);
        for (int it=0; it<nit; it++) {
            init_pn<<<grid, block>>>(p, pn);
            cudaDeviceSynchronize();
            update_p<<<grid, block>>>(p, pn, b, dx, dy);
            cudaDeviceSynchronize();
        }
        gettimeofday(&et2, NULL);

        /* update u, v */
        gettimeofday(&st3, NULL);
        init_unvn<<<grid, block>>>(u, un, v, vn);
        cudaDeviceSynchronize();
        update_uv<<<grid, block>>>(u, un, v, vn, p, rho, dx, dy, dt);
        cudaDeviceSynchronize();
        gettimeofday(&et3, NULL);

        /* show runtime */
        t1 = time_diff_sec(st1, et1);
        t2 = time_diff_sec(st2, et2);
        t3 = time_diff_sec(st3, et3);
        printf("step=%d: %lf us\n",n,t1);
        printf("step=%d: %lf us\n",n,t2);
        printf("step=%d: %lf us (%lf GFlops)\n",n,t3,2.*n*n*n/(t3/1e6)/1e9);
    }

    /* check results */
    printf("\n*** Check results ***");
    printf("\nb[:-3][:-3]=\n");
    for (int i=nx-3; i<nx; i++) {
        for (int j=ny-3; j<ny; j++) {
            printf("%f ", b[j*nx+i]);
        }
        printf("\n");
    }
    printf("\np[:-3][:-3]=\n");
    for (int i=nx-3; i<nx; i++) {
        for (int j=ny-3; j<ny; j++) {
            printf("%f ", p[j*nx+i]);
        }
        printf("\n");
    }
    printf("\nu[:-3][:-3]=\n");
    for (int i=nx-3; i<nx; i++) {
        for (int j=ny-3; j<ny; j++) {
            printf("%f ", u[j*nx+i]);
        }
        printf("\n");
    }
    printf("\nv[:-3][:-3]=\n");
    for (int i=nx-3; i<nx; i++) {
        for (int j=ny-3; j<ny; j++) {
            printf("%f ", v[j*nx+i]);
        }
        printf("\n");
    }

    cudaDeviceSynchronize();
    cudaFree(u); cudaFree(v); cudaFree(b); cudaFree(p);
    cudaFree(un); cudaFree(vn); cudaFree(pn);
}
