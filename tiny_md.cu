#include "core.h"
#include "parameters.h"
#include "wtime.h"

#include <cuda_runtime.h>
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

    // Declarar punteros para la memoria en el host
    float *h_rx, *h_ry, *h_rz, *h_vx, *h_vy, *h_vz, *h_fx, *h_fy, *h_fz;
    // Declarar punteros para la memoria en el dispositivo
    float *d_rx, *d_ry, *d_rz, *d_vx, *d_vy, *d_vz, *d_fx, *d_fy, *d_fz;

    h_rx = (float*)malloc(N * sizeof(float));
    h_ry = (float*)malloc(N * sizeof(float));
    h_rz = (float*)malloc(N * sizeof(float));

    h_vx = (float*)malloc(N * sizeof(float));
    h_vy = (float*)malloc(N * sizeof(float));
    h_vz = (float*)malloc(N * sizeof(float));

    h_fx = (float*)malloc(N * sizeof(float));
    h_fy = (float*)malloc(N * sizeof(float));
    h_fz = (float*)malloc(N * sizeof(float));

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
    init_pos(h_rx, h_ry, h_rz, Rho);

    // Asignar memoria en el dispositivo
    cudaMalloc((void**)&d_rx, N * sizeof(float));
    cudaMalloc((void**)&d_ry, N * sizeof(float));
    cudaMalloc((void**)&d_rz, N * sizeof(float));
    cudaMalloc((void**)&d_vx, N * sizeof(float));
    cudaMalloc((void**)&d_vy, N * sizeof(float));
    cudaMalloc((void**)&d_vz, N * sizeof(float));
    cudaMalloc((void**)&d_fx, N * sizeof(float));
    cudaMalloc((void**)&d_fy, N * sizeof(float));
    cudaMalloc((void**)&d_fz, N * sizeof(float));

    // double start = wtime();
    for (int m = 0; m < 9; m++) {
        Rhob = Rho;
        Rho = RHOI - 0.1 * (float)m;
        cell_V = (float)N / Rho;
        cell_L = cbrtf(cell_V);
        tail = 16.0 * M_PI * Rho * ((2.0 / 3.0) * pow(RCUT, -9) - pow(RCUT, -3)) / 3.0;
        Etail = tail * (float)N;
        Ptail = tail * Rho;

        sf = cbrtf(Rhob / Rho);
        for (int k = 0; k < N; k++) { // reescaleo posiciones a nueva densidad
            h_rx[k] *= sf;
            h_ry[k] *= sf;
            h_rz[k] *= sf;
        }
        init_vel(h_vx, h_vy, h_vz, &Temp, &Ekin);

        // Copiar posiciones a la GPU
        cudaMemcpy(d_rx, h_rx, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ry, h_ry, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rz, h_rz, N * sizeof(float), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize(); // Esperar a que se complete la inicialización de velocidades

        forces<<<(N + 255) / 256, 256>>>(d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, &Epot, &Pres, &Temp, Rho, cell_V, cell_L);
        cudaDeviceSynchronize(); // Esperar a que se complete el cálculo de fuerzas

        for (int i = 1; i < TEQ; i++) { // loop de equilibracion
            velocity_verlet<<<(N + 255) / 256, 256>>>(d_rx, d_ry, d_rz, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, &Epot, &Ekin, &Pres, &Temp, Rho, cell_V, cell_L);
            cudaDeviceSynchronize(); // Esperar a que se complete la actualización de posiciones y velocidades

            sf = sqrtf(T0 / Temp);
            for (int k = 0; k < N; k++) { // reescaleo de velocidades
                h_vx[k] *= sf;
                h_vy[k] *= sf;
                h_vz[k] *= sf;
            }
            // Copiar velocidades reescaladas de vuelta a la GPU
            cudaMemcpy(d_vx, h_vx, N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_vy, h_vy, N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_vz, h_vz, N * sizeof(float), cudaMemcpyHostToDevice);
        }

        int mes = 0;
        float epotm = 0.0, presm = 0.0;
        for (int i = TEQ; i < TRUN; i++) { // loop de medicion
            velocity_verlet<<<(N + 255) / 256, 256>>>(d_rx, d_ry, d_rz, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, &Epot, &Ekin, &Pres, &Temp, Rho, cell_V, cell_L);
            cudaDeviceSynchronize(); // Esperar a que se complete la actualización de posiciones y velocidades

            sf = sqrtf(T0 / Temp);
            for (int k = 0; k < N; k++) { // reescaleo de velocidades
                h_vx[k] *= sf;
                h_vy[k] *= sf;
                h_vz[k] *= sf;
            }
            // Copiar velocidades reescaladas de vuelta a la GPU
            cudaMemcpy(d_vx, h_vx, N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_vy, h_vy, N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_vz, h_vz, N * sizeof(float), cudaMemcpyHostToDevice);

            if (i % TMES == 0) {
                Epot += Etail;
                Pres += Ptail;

                epotm += Epot;
                presm += Pres;
                mes++;

                fprintf(file_thermo, "%f %f %f %f %f\n", t, Temp, Pres, Epot, Epot + Ekin);
                fprintf(file_xyz, "%d\n\n", N);
                for (int k = 0; k < N; k++) {
                    fprintf(file_xyz, "Ar %e %e %e\n", h_rx[k], h_ry[k], h_rz[k]);
                }
            }

            t += DT;
        }
        printf("%f\t%f\t%f\t%f\n", Rho, cell_V, epotm / (double)mes, presm / (double)mes);
    }

    // double elapsed = wtime() - start;
    // printf("# Tiempo total de simulación = %f segundos\n", elapsed);
    printf("# Tiempo simulado = %f [fs]\n", t * 1.6);
    // printf("# ns/day = %f\n", (1.6e-6 * t) / elapsed * 86400);
    // printf("# cells/s = %f\n\n", (N / (elapsed)));

    // Cierre de archivos
    fclose(file_thermo);
    fclose(file_xyz);

    // Liberación de memoria
    free(h_rx);
    free(h_ry);
    free(h_rz);
    free(h_fx);
    free(h_fy);
    free(h_fz);
    free(h_vx);
    free(h_vy);
    free(h_vz);

    // Liberar memoria en el dispositivo
    cudaFree(d_rx);
    cudaFree(d_ry);
    cudaFree(d_rz);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    return 0;
}
