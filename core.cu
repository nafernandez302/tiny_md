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
    const float* rx, const float* ry, const float* rz,
    float* fx, float* fy, float* fz,
    float* epot, float* pres, float* temp, float rho, float V, float L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N - 1)
        return;

    float local_epot = 0.0f;
    float pres_vir = 0.0f;
    const float rcut2 = RCUT * RCUT;
    for (int i = 0; i < (N - 1); i += 1) {
        float xi = rx[i];
        float yi = ry[i];
        float zi = rz[i];
        for (int j = i + 1; j < N; j += 1) {
            const float xj = rx[j];
            const float yj = ry[j];
            const float zj = rz[j];

            // distancia mínima entre r_i y r_j
            float _rx = xi - xj;
            _rx = minimum_image(_rx, L);
            float _ry = yi - yj;
            _ry = minimum_image(_ry, L);
            float _rz = zi - zj;
            _rz = minimum_image(_rz, L);

            const float rij2 = _rx * _rx + _ry * _ry + _rz * _rz;

            if (rij2 <= rcut2) {
                const float r2inv = 1.0f / rij2;
                const float r6inv = r2inv * r2inv * r2inv;

                float fr = 24.0f * r2inv * r6inv * (2.0f * r6inv - 1.0f);

                // --RACECONDITION INIT--
                atomicAdd(&fx[i], fr * _rx);
                atomicAdd(&fy[i], fr * _ry);
                atomicAdd(&fz[i], fr * _rz);

                atomicAdd(&fx[j], -fr * _rx);
                atomicAdd(&fy[j], -fr * _ry);
                atomicAdd(&fz[j], -fr * _rz);

                local_epot += (4.0f * r6inv * (r6inv - 1.0f) - ECUT);
                pres_vir += fr * rij2;

                // --RACECONDITION END--
            }
        }
    }
    pres_vir /= (V * 3.0f);
    *pres = *temp * rho + pres_vir;
    *epot = local_epot;
}

__host__ void init_vel(float* vx, float* vy, float* vz, float* temp, float* ekin)
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

__host__ float pbc(float cordi, const float cell_length)
{
    // condiciones periodicas de contorno coordenadas entre [0,L)
    if (cordi <= 0.0f) {
        cordi += cell_length;
    } else if (cordi > cell_length) {
        cordi -= cell_length;
    }
    return cordi;
}

__host__ void velocity_verlet(float* rx, float* ry, float* rz, float* vx, float* vy, float* vz,
                              float* fx, float* fy, float* fz, float* epot,
                              float* ekin, float* pres, float* temp, const float rho,
                              const float V, const float L)
{
    float *d_rx, *d_ry, *d_rz, *d_vx, *d_vy, *d_vz, *d_fx, *d_fy, *d_fz, *d_epot;
    float *d_Epot, *d_Ekin, *d_Temp, *d_Pres;
    cudaMalloc(&d_Epot, sizeof(float));
    cudaMalloc(&d_Ekin, sizeof(float));
    cudaMalloc(&d_Temp, sizeof(float));
    cudaMalloc(&d_Pres, sizeof(float));

    for (int i = 0; i < N; i += 1) {
        rx[i] += vx[i] * DT + 0.5f * fx[i] * DT * DT;
        ry[i] += vy[i] * DT + 0.5f * fy[i] * DT * DT;
        rz[i] += vz[i] * DT + 0.5f * fz[i] * DT * DT;

        // Aplicar condiciones de frontera periódicas
        rx[i] = pbc(rx[i], L);
        ry[i] = pbc(ry[i], L);
        rz[i] = pbc(rz[i], L);

        vx[i] += 0.5f * fx[i] * DT;
        vy[i] += 0.5f * fy[i] * DT;
        vz[i] += 0.5f * fz[i] * DT;
    }

    cudaMalloc(&d_rx, N * sizeof(float));
    cudaMalloc(&d_ry, N * sizeof(float));
    cudaMalloc(&d_rz, N * sizeof(float));
    cudaMalloc(&d_vx, N * sizeof(float));
    cudaMalloc(&d_vy, N * sizeof(float));
    cudaMalloc(&d_vz, N * sizeof(float));
    cudaMalloc(&d_fx, N * sizeof(float));
    cudaMalloc(&d_fy, N * sizeof(float));
    cudaMalloc(&d_fz, N * sizeof(float));
    cudaMemcpy(d_Temp, temp, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Epot, epot, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pres, pres, sizeof(float), cudaMemcpyHostToDevice);
    // Copiar posiciones a la GPU
    cudaMemcpy(d_rx, rx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ry, ry, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rz, rz, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_vx, vx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, vy, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, vz, N * sizeof(float), cudaMemcpyHostToDevice);


    cudaDeviceSynchronize(); // Esperar a que se complete la inicialización de velocidades

    cudaMemset(d_fx, 0, N * sizeof(float));
    cudaMemset(d_fy, 0, N * sizeof(float));
    cudaMemset(d_fz, 0, N * sizeof(float));
    forces<<<(N + 255) / 256, 256>>>(d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, d_Epot, d_Pres, d_Temp, rho, V, L);
    cudaDeviceSynchronize();

    cudaMemcpy(epot, d_Epot, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pres, d_Pres, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp, d_Temp, sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(fx, d_fx, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fy, d_fy, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fz, d_fz, N * sizeof(float), cudaMemcpyDeviceToHost);

    float sumv2 = 0.0;
    // #pragma omp parallel for reduction(+:sumv2)
    for (int i = 0; i < N; i += 1) { // actualizo velocidades
        vx[i] += 0.5f * fx[i] * DT;
        vy[i] += 0.5f * fy[i] * DT;
        vz[i] += 0.5f * fz[i] * DT;

        sumv2 += vx[i] * vx[i] + vy[i] * vy[i]
            + vz[i] * vz[i];
    }

    *ekin = 0.5 * sumv2;
    *temp = sumv2 / (3.0f * N);
}

// Función para inicializar posiciones (puedes adaptarla a CUDA si es necesario)
__host__ void init_pos(float* rx, float* ry, float* rz, const float rho)
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