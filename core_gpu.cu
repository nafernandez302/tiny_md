#include "parameters.h"
#include <cuda_runtime.h>
#include <math.h>

#define ECUT (4.0f * (powf(RCUT, -12.0f) - powf(RCUT, -6.0f)))

// Kernel de CUDA: un hilo por partícula `i`
__global__ void forces_kernel(
    const float* rx, const float* ry, const float* rz,
    float* fx, float* fy, float* fz,
    float* epot_accum, float* pres_accum,
    float rho, float V, float L, int nParticles)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float rcut2 = RCUT * RCUT;
    float local_epot = 0.0f;
    float local_pres = 0.0f;

    float xi = rx[i], yi = ry[i], zi = rz[i];
    float fxi = 0, fyi = 0, fzi = 0;

    // interacción con j > i
    for (int j = i + 1; j < N; ++j) {
        float _rx = xi - rx[j];
        if (_rx <= -0.5f * L) _rx += L;
        else if (_rx > 0.5f * L) _rx -= L;
        float _ry = yi - ry[j];
        if (_ry <= -0.5f * L) _ry += L;
        else if (_ry > 0.5f * L) _ry -= L;
        float _rz = zi - rz[j];
        if (_rz <= -0.5f * L) _rz += L;
        else if (_rz > 0.5f * L) _rz -= L;

        float rij2 = _rx*_rx + _ry*_ry + _rz*_rz;
        if (rij2 <= rcut2) {
            float r2inv = 1.0f / rij2;
            float r6inv = r2inv * r2inv * r2inv;
            float fr = 24.0f * r2inv * r6inv * (2.0f * r6inv - 1.0f);

            // acumula fuerzas
            fxi += fr * _rx;   fyi += fr * _ry;   fzi += fr * _rz;
            atomicAdd(&fx[j], -fr * _rx);
            atomicAdd(&fy[j], -fr * _ry);
            atomicAdd(&fz[j], -fr * _rz);

            // energía y presión virial
            local_epot += (4.0f * r6inv * (r6inv - 1.0f) - ECUT);
            local_pres += fr * rij2;
        }
    }

    // escribir fuerzas locales
    atomicAdd(&fx[i], fxi);
    atomicAdd(&fy[i], fyi);
    atomicAdd(&fz[i], fzi);

    // acumuladores globales
    atomicAdd(epot_accum, local_epot);
    atomicAdd(pres_accum, local_pres);
}

// Wrapper C que se enlaza desde core.c
extern "C" void forces_cu(
    const float* h_rx, const float* h_ry, const float* h_rz,
    float* h_fx, float* h_fy, float* h_fz,
    float* h_epot, float* h_pres,
    const float* h_temp,    // ← const aquí también
    float rho,
    float V, float L)
{
    const int nParticles = N; // de parameters.h

    // 1) Reservar memoria en device
    float *d_rx, *d_ry, *d_rz, *d_fx, *d_fy, *d_fz;
    float *d_epot, *d_pres;
    cudaMalloc(&d_rx, N*sizeof(float));
    cudaMalloc(&d_ry, N*sizeof(float));
    cudaMalloc(&d_rz, N*sizeof(float));
    cudaMalloc(&d_fx, N*sizeof(float));
    cudaMalloc(&d_fy, N*sizeof(float));
    cudaMalloc(&d_fz, N*sizeof(float));
    cudaMalloc(&d_epot, sizeof(float));
    cudaMalloc(&d_pres, sizeof(float));

    // 2) Copiar posiciones y temp (no usada en kernel, pero la dejas si quieres)
    cudaMemcpy(d_rx, h_rx, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ry, h_ry, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rz, h_rz, N*sizeof(float), cudaMemcpyHostToDevice);

    // 3) Inicializar fuerzas y acumuladores a cero
    cudaMemset(d_fx,  0, N*sizeof(float));
    cudaMemset(d_fy,  0, N*sizeof(float));
    cudaMemset(d_fz,  0, N*sizeof(float));
    cudaMemset(d_epot, 0, sizeof(float));
    cudaMemset(d_pres, 0, sizeof(float));

    // 4) Lanzar kernel
    int blockSize = 128;
    int gridSize  = (N + blockSize - 1) / blockSize;
    forces_kernel<<<gridSize,blockSize>>>(
        d_rx, d_ry, d_rz,
        d_fx, d_fy, d_fz,
        d_epot, d_pres,
        rho, V, L, nParticles);
    cudaDeviceSynchronize();

    // 5) Copiar resultados de vuelta
    cudaMemcpy(h_fx,    d_fx,    N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy,    d_fy,    N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz,    d_fz,    N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_epot,  d_epot,  sizeof(float),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pres,  d_pres,  sizeof(float),   cudaMemcpyDeviceToHost);

    // 6) Liberar
    cudaFree(d_rx);  cudaFree(d_ry);  cudaFree(d_rz);
    cudaFree(d_fx);  cudaFree(d_fy);  cudaFree(d_fz);
    cudaFree(d_epot); cudaFree(d_pres);
}
