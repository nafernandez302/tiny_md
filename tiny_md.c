#define _XOPEN_SOURCE 500  // M_PI
#include "core.h"
#include "parameters.h"
#include "wtime.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>


int main()
{
    FILE *file_xyz, *file_thermo;
    file_xyz = fopen("trajectory.xyz", "w");
    file_thermo = fopen("thermo.log", "w");
    float Ekin, Epot, Temp, Pres; // variables macroscopicas
    float Rho, cell_V, cell_L, tail, Etail, Ptail;
    float *rx, *ry, *rz, *vx, *vy, *vz, *fx, *fy, *fz; // variables microscopicas

    rx = (float*)malloc(N * sizeof(float));
    ry = (float*)malloc(N * sizeof(float));
    rz = (float*)malloc(N * sizeof(float));

    vx = (float*)malloc(N * sizeof(float));
    vy = (float*)malloc(N * sizeof(float));
    vz = (float*)malloc(N * sizeof(float));

    fx = (float*)malloc(N * sizeof(float));
    fy = (float*)malloc(N * sizeof(float));
    fz = (float*)malloc(N * sizeof(float));

    printf("# Número de partículas:      %d\n", N);
    printf("# Temperatura de referencia: %.2f\n", T0);
    printf("# Pasos de equilibración:    %d\n", TEQ);
    printf("# Pasos de medición:         %d\n", TRUN - TEQ);
    printf("# (mediciones cada %d pasos)\n", TMES);
    printf("# densidad, volumen, energía potencial media, presión media\n");
    fprintf(file_thermo, "# t Temp Pres Epot Etot\n");

    srand(SEED);
    float t = 0.0, sf;
    float Rhob;
    Rho = RHOI;
    init_pos(rx, ry, rz, Rho);
    double start = wtime();
    for (int m = 0; m < 9; m++) {
        Rhob = Rho;
        Rho = RHOI - 0.1 * (float)m;
        cell_V = (float)N / Rho;
        cell_L = cbrtf(cell_V);
        tail = 16.0 * M_PI * Rho * ((2.0 / 3.0) * pow(RCUT, -9) - pow(RCUT, -3)) / 3.0;
        Etail = tail * (float)N;
        Ptail = tail * Rho;

        int i = 0;
        sf = cbrtf(Rhob / Rho);
        for (int k = 0; k < N; k++) { // reescaleo posiciones a nueva densidad
            rx[k] *= sf;
            ry[k] *= sf;
            rz[k] *= sf;
        }
        init_vel(vx, vy, vz, &Temp, &Ekin);
        forces(rx, ry, rz, fx, fy, fz, &Epot, &Pres, &Temp, Rho, cell_V, cell_L);

        for (i = 1; i < TEQ; i++) { // loop de equilibracion

            velocity_verlet(rx, ry, rz, vx, vy, vz, fx, fy, fz, &Epot, &Ekin, &Pres, &Temp, Rho, cell_V, cell_L);

            sf = sqrtf(T0 / Temp);
            for (int k = 0; k < N; k++) { // reescaleo de velocidades
                vx[k] *= sf;
                vy[k] *= sf;
                vz[k] *= sf;
            }
        }

        int mes = 0;
        float epotm = 0.0, presm = 0.0;
        for (i = TEQ; i < TRUN; i++) { // loop de medicion

            velocity_verlet(rx, ry, rz, vx, vy, vz, fx, fy, fz, &Epot, &Ekin, &Pres, &Temp, Rho, cell_V, cell_L);

            sf = sqrtf(T0 / Temp);
            for (int k = 0; k < N; k++) { // reescaleo de velocidades
                vx[k] *= sf;
                vy[k] *= sf;
                vz[k] *= sf;
            }

            if (i % TMES == 0) {
                Epot += Etail;
                Pres += Ptail;

                epotm += Epot;
                presm += Pres;
                mes++;

                fprintf(file_thermo, "%f %f %f %f %f\n", t, Temp, Pres, Epot, Epot + Ekin);
                fprintf(file_xyz, "%d\n\n", N);
                for (int k = 0; k < N; k++) {
                    fprintf(file_xyz, "Ar %e %e %e\n", rx[k], ry[k], rz[k]);
                }
            }

            t += DT;
        }
        printf("%f\t%f\t%f\t%f\n", Rho, cell_V, epotm / (double)mes, presm / (double)mes);
    }

    double elapsed = wtime() - start;
    printf("# Tiempo total de simulación = %f segundos\n", elapsed);
    printf("# Tiempo simulado = %f [fs]\n", t * 1.6);
    printf("# ns/day = %f\n", (1.6e-6 * t) / elapsed * 86400);
    //                       ^1.6 fs -> ns       ^sec -> day
    printf("# cells/s = %f\n\n", (N /(elapsed)));

    // Cierre de archivos
    fclose(file_thermo);
    fclose(file_xyz);

    // Liberacion de memoria
    free(rx);
    free(ry);
    free(rz);
    free(fx);
    free(fy);
    free(fz);
    free(vx);
    free(vy);
    free(vz);
    return 0;
}
