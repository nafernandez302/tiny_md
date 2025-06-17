#include "core.h"
#include "parameters.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h> // rand()


#include <cuda_runtime.h>

#define ECUT (4.0f * (powf(RCUT, -12.0f) - powf(RCUT, -6.0f)))


__device__ float minimum_image(float cordi, const float cell_length)
{
    float caso_sumar = (cordi <= -0.5f * cell_length) ? 1.0f : 0.0f;
    float caso_restar = (cordi > 0.5f * cell_length) ? 1.0f : 0.0f;
    cordi += (caso_sumar)*cell_length - (caso_restar)*cell_length;
    return cordi;
}

__global__ void forces(
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    const float* __restrict__ rz,
    float* fx,
    float* fy,
    float* fz,
    float* epot,
    float* pres,
    float* temp,
    float rho,
    float V,
    float L)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float rcut2 = RCUT * RCUT;
    float xi = rx[i];
    float yi = ry[i];
    float zi = rz[i];

    float local_epot = 0.0f;
    float pres_vir   = 0.0f;

    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int j = i + 1; j < N; j += stride) {
        float dx = xi - rx[j];
        float dy = yi - ry[j];
        float dz = zi - rz[j];

        dx = minimum_image(dx, L);
        dy = minimum_image(dy, L);
        dz = minimum_image(dz, L);

        float rij2 = dx*dx + dy*dy + dz*dz;
        if (rij2 <= rcut2) {
            float r2inv = 1.0f / rij2;
            float r6inv = r2inv * r2inv * r2inv;
            float fr = 24.0f * r2inv * r6inv * (2.0f * r6inv - 1.0f);

            float fxij = fr * dx;
            float fyij = fr * dy;
            float fzij = fr * dz;

            atomicAdd(&fx[i],  fxij);
            atomicAdd(&fy[i],  fyij);
            atomicAdd(&fz[i],  fzij);
            atomicAdd(&fx[j], -fxij);
            atomicAdd(&fy[j], -fyij);
            atomicAdd(&fz[j], -fzij);

            local_epot += (4.0f * r6inv * (r6inv - 1.0f) - ECUT);
            pres_vir   += fr * rij2;
        }
    }

    atomicAdd(epot, local_epot);
    atomicAdd(pres, pres_vir);
}

 void init_vel(float* vx, float* vy, float* vz, float* temp, float* ekin)
{

    float sf, sumvx = 0.0f, sumvy = 0.0f, sumvz = 0.0f, sumv2 = 0.0f;

    for (int i = 0; i < N; i += 1) {
        vx[i] = rand() / (float)RAND_MAX - 0.5f;
        vy[i] = rand() / (float)RAND_MAX - 0.5f;
        vz[i] = rand() / (float)RAND_MAX - 0.5f;

        sumvx += vx[i];
        sumvy += vy[i];
        sumvz += vz[i];
        sumv2 += vx[i] * vx[i] + vy[i] * vy[i]
            + vz[i] * vz[i];
    }

    sumvx /= (float)N;
    sumvy /= (float)N;
    sumvz /= (float)N;
    *temp = sumv2 / (3.0f * N);
    *ekin = 0.5f * sumv2;
    sf = sqrtf(T0 / *temp);

    for (int i = 0; i < N; i += 1) { // elimina la velocidad del centro de masa
        // y ajusta la temperatura
        vx[i] = (vx[i] - sumvx) * sf;
        vy[i] = (vy[i] - sumvy) * sf;
        vz[i] = (vz[i] - sumvz) * sf;
    }
}

 float pbc(float cordi, const float cell_length)
{
    // condiciones periodicas de contorno coordenadas entre [0,L)
    if (cordi <= 0.0f) {
        cordi += cell_length;
    } else if (cordi > cell_length) {
        cordi -= cell_length;
    }
    return cordi;
}


void velocity_verlet(
    float* rx,  float* ry,  float* rz,
    float* vx,  float* vy,  float* vz,
    float* fx,  float* fy,  float* fz,
    float* epot, float* ekin, float* pres, float* temp,
    const float rho, const float V, const float L)
{
    // 1) Primer medio paso (posiciones + medio impulso) en HOST
    for (int i = 0; i < N; i++) {
        rx[i] += vx[i] * DT + 0.5f * fx[i] * DT * DT;
        ry[i] += vy[i] * DT + 0.5f * fy[i] * DT * DT;
        rz[i] += vz[i] * DT + 0.5f * fz[i] * DT * DT;

        rx[i] = pbc(rx[i], L);
        ry[i] = pbc(ry[i], L);
        rz[i] = pbc(rz[i], L);

        vx[i] += 0.5f * fx[i] * DT;
        vy[i] += 0.5f * fy[i] * DT;
        vz[i] += 0.5f * fz[i] * DT;
    }

    // 2) Reserva y copia datos a GPU
    float *d_rx, *d_ry, *d_rz;
    float *d_fx, *d_fy, *d_fz;
    float *d_epot, *d_pres, *d_temp;

    cudaMalloc(&d_rx,   N*sizeof(float));
    cudaMalloc(&d_ry,   N*sizeof(float));
    cudaMalloc(&d_rz,   N*sizeof(float));
    cudaMalloc(&d_fx,   N*sizeof(float));
    cudaMalloc(&d_fy,   N*sizeof(float));
    cudaMalloc(&d_fz,   N*sizeof(float));
    cudaMalloc(&d_epot, sizeof(float));
    cudaMalloc(&d_pres, sizeof(float));
    cudaMalloc(&d_temp, sizeof(float));

    // copia posiciones y temp
    cudaMemcpy(d_rx, rx, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ry, ry, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rz, rz, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp, temp, sizeof(float), cudaMemcpyHostToDevice);

    // 3) Inicializa arrays de fuerzas y acumuladores
    cudaMemset(d_fx,   0, N*sizeof(float));
    cudaMemset(d_fy,   0, N*sizeof(float));
    cudaMemset(d_fz,   0, N*sizeof(float));
    cudaMemset(d_epot, 0, sizeof(float));
    cudaMemset(d_pres, 0, sizeof(float));

    // 4) Lanza kernel CUDA
    int threads = 128;
    int blocks  = (N + threads - 1) / threads;
    forces<<<blocks, threads>>>(
        d_rx, d_ry, d_rz,
        d_fx, d_fy, d_fz,
        d_epot, d_pres, d_temp,
        rho, V, L
    );
    cudaDeviceSynchronize();

    // 5) Recupera fuerzas y acumuladores al HOST
    cudaMemcpy(fx,   d_fx,   N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fy,   d_fy,   N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fz,   d_fz,   N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(epot, d_epot, sizeof(float),   cudaMemcpyDeviceToHost);
    cudaMemcpy(pres, d_pres, sizeof(float),   cudaMemcpyDeviceToHost);

    // 6) Libera memoria GPU
    cudaFree(d_rx);
    cudaFree(d_ry);
    cudaFree(d_rz);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
    cudaFree(d_epot);
    cudaFree(d_pres);
    cudaFree(d_temp);

    // 7) Segundo medio paso de velocidades y cálculo de energía cinética
    float sumv2 = 0.0f;
    for (int i = 0; i < N; i++) {
        vx[i] += 0.5f * fx[i] * DT;
        vy[i] += 0.5f * fy[i] * DT;
        vz[i] += 0.5f * fz[i] * DT;
        sumv2 += vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
    }
    *ekin = 0.5f * sumv2;
    *temp = sumv2 / (3.0f * N);

    // 8) Escala y completa presión igual que en CPU
    *pres = (*temp) * rho + (*pres) / (3.0f * V);
}

// Función para inicializar posiciones (puedes adaptarla a CUDA si es necesario)
 void init_pos(float* rx, float* ry, float* rz, const float rho)
{
    // inicialización de las posiciones de los átomos en un cristal FCC

    const float a = cbrtf(4.0f / rho);
    const int nucells = round(cbrtf((float)N / 4.0f));
    int idx = 0;
    float fi = 0.0f;
    for (int i = 0; i < nucells; i++, fi += 1.0f) {
        float fj = 0.0f;
        for (int j = 0; j < nucells; j++, fj += 1.0f) {
            float fk = 0.0f;
            for (int k = 0; k < nucells; k++, fk += 1.0f) {
                rx[idx] = fi * a; // x
                ry[idx] = fj * a; // y
                rz[idx] = fk * a; // z
                                  // del mismo átomo
                rx[idx + 1] = (fi + 0.5f) * a;
                ry[idx + 1] = (fj + 0.5f) * a;
                rz[idx + 1] = fk * a;

                rx[idx + 2] = (fi + 0.5f) * a;
                ry[idx + 2] = fj * a;
                rz[idx + 2] = (fk + 0.5f) * a;

                rx[idx + 3] = fi * a;
                ry[idx + 3] = (fj + 0.5f) * a;
                rz[idx + 3] = (fk + 0.5f) * a;

                idx += 4;
            }
        }
    }
}


// Función para aplicar condiciones de frontera periódicas