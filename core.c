#include "core.h"
#include "parameters.h"
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h> // rand()

#define ECUT (4.0f * (powf(RCUT, -12.0f) - powf(RCUT, -6.0f)))


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


static float minimum_image(float cordi, const float cell_length)
{
    // imagen más cercana
    
    float caso_sumar = (cordi <= -0.5f * cell_length) ? 1.0f : 0.0f;
    float caso_restar = (cordi > 0.5f *  cell_length) ? 1.0f : 0.0f;
    cordi += (caso_sumar) * cell_length
            -(caso_restar) * cell_length;
    return cordi;
}


void forces(const float* rx, const float* ry, const float* rz, 
            float* fx, float* fy, float* fz, float* epot, float* pres,
            const float* temp, const float rho, const float V, const float L)
{
    // calcula las fuerzas LJ (12-6)

    float local_epot = 0.0f;
    float pres_vir = 0.0f;
    const float rcut2 = RCUT * RCUT;
    for (int i = 0; i < N; i++) {
        fx[i] = 0.0;
        fy[i] = 0.0;
        fz[i] = 0.0;
    }        
    #pragma omp parallel for reduction(+: fx[:N], fy[:N], fz[:N], local_epot, pres_vir)
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

                fx[i] += fr * _rx;
                fy[i] += fr * _ry;
                fz[i] += fr * _rz;

                fx[j] -= fr * _rx;
                fy[j] -= fr * _ry;
                fz[j] -= fr * _rz;

                local_epot += (4.0f * r6inv * (r6inv - 1.0f) - ECUT) ;
                pres_vir += fr * rij2;
            }
        }
    }
    pres_vir /= (V * 3.0f);
    *pres = *temp * rho + pres_vir;
    *epot = local_epot;
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
