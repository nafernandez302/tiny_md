#include "core.h"
#include "parameters.h"
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h> // rand()


#include <cuda_runtime.h>


__device__ float minimum_image(float cordi, const float cell_length) {
    float caso_sumar = (cordi <= -0.5f * cell_length) ? 1.0f : 0.0f;
    float caso_restar = (cordi > 0.5f * cell_length) ? 1.0f : 0.0f;
    cordi += (caso_sumar) * cell_length - (caso_restar) * cell_length;
    return cordi;
}

__global__ void compute_forces(
    const float* rx, const float* ry, const float* rz,
    float* fx, float* fy, float* fz,
    float* epot, float* pres_vir,
    int N, float L, float V
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N - 1) return;

    float xi = rx[i], yi = ry[i], zi = rz[i];
    float fx_i = 0.0f, fy_i = 0.0f, fz_i = 0.0f;

    float epot_local = 0.0f;
    float virial_local = 0.0f;
    const float rcut2 = RCUT * RCUT;

    for (int j = i + 1; j < N; j++) {
        float dx = minimum_image(xi - rx[j], L);
        float dy = minimum_image(yi - ry[j], L);
        float dz = minimum_image(zi - rz[j], L);
        float rij2 = dx*dx + dy*dy + dz*dz;

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
    atomicAdd(pres_vir, virial_local);
}

__global__ void init_vel(float* vx, float* vy, float* vz, float* temp, float* ekin) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        vx[idx] = rand() / (float)RAND_MAX - 0.5f;
        vy[idx] = rand() / (float)RAND_MAX - 0.5f;
        vz[idx] = rand() / (float)RAND_MAX - 0.5f;
    }
}

__global__ void velocity_verlet(float* rx, float* ry, float* rz, float* vx, float* vy, float* vz,
                                float* fx, float* fy, float* fz, float* epot,
                                float* ekin, float* pres, float* temp, const float rho,
                                const float V, const float L) {
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
__global__ void init_pos(float* rx, float* ry, float* rz, const float rho) {
    // Implementación de la inicialización de posiciones en CUDA
    // ...
}

// Función para aplicar condiciones de frontera periódicas
__device__ float pbc(float cordi, const float cell_length) {
    if (cordi <= 0.0f) {
        cordi += cell_length;
    } else if (cordi > cell_length) {
        cordi -= cell_length;
    }
    return cordi;
}
