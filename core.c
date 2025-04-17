#include "core.h"
#include "parameters.h"

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


void init_vel(float* vxyz, float* temp, float* ekin)
{
    // inicialización de velocidades aleatorias

    float sf, sumvx = 0.0f, sumvy = 0.0f, sumvz = 0.0f, sumv2 = 0.0f;

    for (int i = 0; i < 3 * N; i += 3) {
        vxyz[i + 0] = rand() / (float)RAND_MAX - 0.5f;
        vxyz[i + 1] = rand() / (float)RAND_MAX - 0.5f;
        vxyz[i + 2] = rand() / (float)RAND_MAX - 0.5f;

        sumvx += vxyz[i + 0];
        sumvy += vxyz[i + 1];
        sumvz += vxyz[i + 2];
        sumv2 += vxyz[i + 0] * vxyz[i + 0] + vxyz[i + 1] * vxyz[i + 1]
            + vxyz[i + 2] * vxyz[i + 2];
    }

    sumvx /= (float)N;
    sumvy /= (float)N;
    sumvz /= (float)N;
    *temp = sumv2 / (3.0f * N);
    *ekin = 0.5f * sumv2;
    sf = sqrtf(T0 / *temp);

    for (int i = 0; i < 3 * N; i += 3) { // elimina la velocidad del centro de masa
        // y ajusta la temperatura
        vxyz[i + 0] = (vxyz[i + 0] - sumvx) * sf;
        vxyz[i + 1] = (vxyz[i + 1] - sumvy) * sf;
        vxyz[i + 2] = (vxyz[i + 2] - sumvz) * sf;
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


void forces(const float* rx, const float* ry, const float* rz, float* fxyz, float* epot, float* pres,
            const float* temp, const float rho, const float V, const float L)
{
    // calcula las fuerzas LJ (12-6)

    for (int i = 0; i < 3 * N; i++) {
        fxyz[i] = 0.0;
    }
    float pres_vir = 0.0f;
    const float rcut2 = RCUT * RCUT;
    *epot = 0.0f;

    for (int i = 0; i < 3 * (N - 1); i += 3) {
        int i_MOD = (i / 3) % (N-1);
        float xi = rx[i_MOD];
        float yi = ry[i_MOD];
        float zi = rz[i_MOD];

        for (int j = i + 3; j < 3 * N; j += 3) {
            int j_MOD = (j /3) % N;
            const float xj = rx[j_MOD];
            const float yj = ry[j_MOD];
            const float zj = rz[j_MOD];

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

                fxyz[i + 0] += fr * _rx;
                fxyz[i + 1] += fr * _ry;
                fxyz[i + 2] += fr * _rz;

                fxyz[j + 0] -= fr * _rx;
                fxyz[j + 1] -= fr * _ry;
                fxyz[j + 2] -= fr * _rz;

                *epot += (4.0f * r6inv * (r6inv - 1.0f) - ECUT) ;
                pres_vir += fr * rij2;
            }
        }
    }
    pres_vir /= (V * 3.0f);
    *pres = *temp * rho + pres_vir;
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


void velocity_verlet(float* rx, float* ry, float* rz, float* vxyz, float* fxyz, float* epot,
                     float* ekin, float* pres, float* temp, const float rho,
                     const float V, const float L)
{

    for (int i = 0; i < 3 * N; i += 3) { // actualizo posiciones
        int i_MOD = (i / 3)%N;
        rx[i_MOD] += vxyz[i + 0] * DT + 0.5f * fxyz[i + 0] * DT * DT;
        ry[i_MOD] += vxyz[i + 1] * DT + 0.5f * fxyz[i + 1] * DT * DT;
        rz[i_MOD] += vxyz[i + 2] * DT + 0.5f * fxyz[i + 2] * DT * DT;

        rx[i_MOD] = pbc(rx[i_MOD], L);
        ry[i_MOD] = pbc(ry[i_MOD], L);
        rz[i_MOD] = pbc(rz[i_MOD], L);

        vxyz[i + 0] += 0.5f * fxyz[i + 0] * DT;
        vxyz[i + 1] += 0.5f * fxyz[i + 1] * DT;
        vxyz[i + 2] += 0.5f * fxyz[i + 2] * DT;
    }

    forces(rx, ry, rz, fxyz, epot, pres, temp, rho, V, L); // actualizo fuerzas

    float sumv2 = 0.0;
    for (int i = 0; i < 3 * N; i += 3) { // actualizo velocidades
        vxyz[i + 0] += 0.5f * fxyz[i + 0] * DT;
        vxyz[i + 1] += 0.5f * fxyz[i + 1] * DT;
        vxyz[i + 2] += 0.5f * fxyz[i + 2] * DT;

        sumv2 += vxyz[i + 0] * vxyz[i + 0] + vxyz[i + 1] * vxyz[i + 1]
            + vxyz[i + 2] * vxyz[i + 2];
    }

    *ekin = 0.5 * sumv2;
    *temp = sumv2 / (3.0f * N);
}
