#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <immintrin.h>
using namespace std;
typedef vector<vector<float>> matrix;

int main() {
    /* variables preparation */
    const int nx = 161; //41
    const int ny = 161; // 41
    int nt = 10; // 500;
    int nit = 50;
    float dx = 2. / (nx - 1);
    float dy = 2. / (ny - 1);
    float dt = 0.005; // 0.01
    float rho = 1;
    float nu = 0.02;

    vector<float> x(nx);
    vector<float> y(ny);
    for (int i=0; i<nx; i++) 
        x[i] = i * dx;
    for (int j=0; j<ny; j++) 
        y[j] = j * dy;

    matrix u(ny, vector<float>(nx,0));
    matrix v(ny, vector<float>(nx,0));
    matrix p(ny, vector<float>(nx,0));
    matrix b(ny, vector<float>(nx,0));

    matrix pn(ny, vector<float>(nx,0));
    matrix un(ny, vector<float>(nx,0));
    matrix vn(ny, vector<float>(nx,0));

    for (int n=0; n<nt; n++) {
        auto tic = chrono::steady_clock::now();
        for (int j=1; j<ny-1; j++) {
            for (int i=1; i<nx-1; i++) {
                    b[j][i] = rho * (1 / dt *
                            ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) -
                            ((u[j][i+1] - u[j][i-1]) / (2 * dx))*((u[j][i+1] - u[j][i-1]) / (2 * dx)) - 
                            2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) *
                            (v[j][i+1] - v[j][i-1]) / (2 * dx)) - 
                            ((v[j+1][i] - v[j-1][i]) / (2 * dy))*((v[j+1][i] - v[j-1][i]) / (2 * dy)));
            }
        }

        // Poisson
        for (int it=0; it<nit; it++) {
            for (int j=0; j<ny; j++) {
                for (int i=0; i<nx; i++) {
                    pn[j][i] = p[j][i];
                }
            }
#pragma omp paralell for
            for (int j=1; j<ny-1; j++) {
                for (int i=1; i<nx-1; i++) {
                    p[j][i] = (dy*dy * (pn[j][i+1] + pn[j][i-1]) +
                            dx*dx * (pn[j+1][i] + pn[j-1][i]) -
                            b[j][i] * dx*dx * dy*dy)/(2 * (dx*dx + dy*dy));
                }
            }

            //境界条件
            for (int j=1; j<ny-1; j++) {
                p[j][nx-1] = p[j][nx-2];
                p[j][0] = p[j][1];
            }
            for (int i=1; i<nx-1;i++) {
                p[0][i] = p[1][i];
                p[ny-1][i] = p[ny-2][i];
            }
        }

        // un, vn
        for (int j=0; j<ny; j++) {
            for (int i=0; i<nx; i++) {
                un[j][i] = u[j][i];
                vn[j][i] = v[j][i];
            }
        }

        // 本番
        for (int j=1; j<ny-1; j++) {
            for (int i=1; i<nx-1; i++) {
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
                                - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
                                - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])
                                + nu * dt / dx*dx * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])
                                + nu * dt / dy*dy * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);
                v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
                                - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
                                - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i])
                                + nu * dt / dx*dx * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])
                                + nu * dt / dy*dy * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
            }
        }
        // u,v 境界
        for (int j=1; j<ny-1; j++) {
            u[j][0] = 0;
            u[j][nx-1] = 0;
            v[j][0] = 0;
            v[j][nx-1] = 0;
        }
        for (int i=1; i<nx-1; i++) {
            u[i][0] = 0;
            u[i][nx-1] = 1;
            v[i][0] = 0;
            v[i][nx-1] = 0;
        }
        auto toc = chrono::steady_clock::now();
        double time = chrono::duration<double>(toc - tic).count();
        printf("N=%d: %lf s (%lf GFlops)\n",n,time,2.*n*n*n/time/1e9);
    }
}
