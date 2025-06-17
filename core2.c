#include "core.h"
#include "parameters.h"
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h> // rand()

#define ECUT (4.0f * (powf(RCUT, -12.0f) - powf(RCUT, -6.0f)))

float *d_rx, *d_ry, *d_rz;
float *d_fx, *d_fy, *d_fz;
float *d_epot, *d_pres;

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


void init_vel(float* vx, float* vy, float* vz, float* temp, float* ekin)
{
    // inicialización de velocidades aleatorias

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


 float minimum_image(float cordi, const float cell_length)
{
    // imagen más cercana
    
    float caso_sumar = (cordi <= -0.5f * cell_length) ? 1.0f : 0.0f;
    float caso_restar = (cordi > 0.5f *  cell_length) ? 1.0f : 0.0f;
    cordi += (caso_sumar) * cell_length
            -(caso_restar) * cell_length;
    return cordi;
}


void forces(const float* rx, const float* ry, const float* rz,
            float* fx, float* fy, float* fz,
            float* epot, float* pres,
            const float* temp,
            float rho,
            float V, float L)
{
    forces_cu(rx, ry, rz, fx, fy, fz, epot, pres, temp, rho, V, L);

    *pres = (*temp) * rho + (*pres) / (V * 3.0f);
}




static float pbc(float cordi, const float cell_length)
{
    // condiciones periodicas de contorno coordenadas entre [0,L)
    if (cordi <= 0.0f) {
        cordi += cell_length;
    } else if (cordi > cell_length) {
        cordi -= cell_length;
    }
    return cordi;
}


void velocity_verlet(float* rx, float* ry, float* rz, float* vx, float* vy, float*vz,
                     float* fx, float* fy, float* fz, float* epot,
                     float* ekin, float* pres, float* temp, const float rho,
                     const float V, const float L)
{

    #pragma omp parallel for
    for (int i = 0; i < N; i += 1) { // actualizo posiciones
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

    forces(rx, ry, rz, fx, fy, fz, epot, pres, temp, rho, V, L); // actualizo fuerzas

    
    float sumv2 = 0.0;
    //#pragma omp parallel for reduction(+:sumv2)
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
