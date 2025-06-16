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

    float xi = rx[i], yi = ry[i], zi = rz[i];
    float fx_i = 0.0f, fy_i = 0.0f, fz_i = 0.0f;

    float epot_local = 0.0f;
    float virial_local = 0.0f;
    const float rcut2 = RCUT * RCUT;

    for (int j = i + 1; j < N; j++) {
        float dx = minimum_image(xi - rx[j], L);
        float dy = minimum_image(yi - ry[j], L);
        float dz = minimum_image(zi - rz[j], L);
        float rij2 = dx * dx + dy * dy + dz * dz;

        if (rij2 < rcut2) {
            float r2inv = 1.0f / rij2;
            float r6inv = r2inv * r2inv * r2inv;
            float fr = 24.0f * r2inv * r6inv * (2.0f * r6inv - 1.0f);

            fx_i += fr * dx;
            fy_i += fr * dy;
            fz_i += fr * dz;

            atomicAdd(&fx[j], -fr * dx);
            atomicAdd(&fy[j], -fr * dy);
            atomicAdd(&fz[j], -fr * dz);

            epot_local += 4.0f * r6inv * (r6inv - 1.0f) - ECUT;
            virial_local += fr * rij2;
        }
    }

    fx[i] += fx_i;
    fy[i] += fy_i;
    fz[i] += fz_i;

    atomicAdd(epot, epot_local);
    atomicAdd(pres, virial_local);
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

__device__ float pbc(float cordi, const float cell_length)
{
    // condiciones periodicas de contorno coordenadas entre [0,L)
    if (cordi <= 0.0f) {
        cordi += cell_length;
    } else if (cordi > cell_length) {
        cordi -= cell_length;
    }
    return cordi;
}

__global__ void velocity_verlet(float* rx, float* ry, float* rz, float* vx, float* vy, float* vz,
                                float* fx, float* fy, float* fz, float* epot,
                                float* ekin, float* pres, float* temp, const float rho,
                                const float V, const float L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
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