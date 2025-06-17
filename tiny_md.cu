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
    float h_Ekin, h_Epot, h_Temp, h_Pres; // variables macroscopicas
    float Rho, cell_V, cell_L, tail, Etail, Ptail;
    dim3 block(32);
    dim3 grid(N);
    // Declarar punteros para la memoria en el host
    float *h_rx, *h_ry, *h_rz, *h_vx, *h_vy, *h_vz, *h_fx, *h_fy, *h_fz;
    // Declarar punteros para la memoria en el dispositivo
    float *d_rx, *d_ry, *d_rz, *d_vx, *d_vy, *d_vz, *d_fx, *d_fy, *d_fz;

    float *d_Epot, *d_Ekin, *d_Temp, *d_Pres;
    cudaMalloc(&d_Epot, sizeof(float));
    cudaMalloc(&d_Ekin, sizeof(float));
    cudaMalloc(&d_Temp, sizeof(float));
    cudaMalloc(&d_Pres, sizeof(float));

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
        init_vel(h_vx, h_vy, h_vz, &h_Temp, &h_Ekin);
        cudaMemcpy(d_Temp, &h_Temp, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Ekin, &h_Ekin, sizeof(float), cudaMemcpyHostToDevice);

        // Copiar posiciones a la GPU
        cudaMemcpy(d_rx, h_rx, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ry, h_ry, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rz, h_rz, N * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(d_vx, h_vx, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vy, h_vy, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vz, h_vz, N * sizeof(float), cudaMemcpyHostToDevice);


        cudaDeviceSynchronize(); // Esperar a que se complete la inicialización de velocidades

        cudaMemset(d_fx, 0, N * sizeof(float));
        cudaMemset(d_fy, 0, N * sizeof(float));
        cudaMemset(d_fz, 0, N * sizeof(float));
        forces<<<grid, block>>>(d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, d_Epot, d_Pres, d_Temp, Rho, cell_V, cell_L);
        cudaDeviceSynchronize(); // Esperar a que se complete el cálculo de fuerzas
        cudaMemcpy(&h_Epot, d_Epot, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_Pres, d_Pres, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_Ekin, d_Ekin, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fx, d_fx, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fy, d_fy, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fz, d_fz, N * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 1; i < TEQ; i++) { // loop de

            velocity_verlet(h_rx, h_ry, h_rz, h_vx, h_vy, h_vz, h_fx, h_fy, h_fz, &h_Epot, &h_Ekin, &h_Pres, &h_Temp, Rho, cell_V, cell_L);

            sf = sqrtf(T0 / h_Temp);
            for (int k = 0; k < N; k++) { // reescaleo de velocidades
                h_vx[k] *= sf;
                h_vy[k] *= sf;
                h_vz[k] *= sf;
            }
        }

        int mes = 0;
        float epotm = 0.0, presm = 0.0;
        for (int i = TEQ; i < TRUN; i++) {
            velocity_verlet(h_rx, h_ry, h_rz, h_vx, h_vy, h_vz,
                            h_fx, h_fy, h_fz, &h_Epot, &h_Ekin, &h_Pres, &h_Temp,
                            Rho, cell_V, cell_L);
            sf = sqrtf(T0 / h_Temp);
            for (int k = 0; k < N; k++) { // reescaleo de velocidades
                h_vx[k] *= sf;
                h_vy[k] *= sf;
                h_vz[k] *= sf;
            }
            // Copiar Temp a host para calcular el factor de escala
            float temp_host;
            cudaMemcpy(&temp_host, d_Temp, sizeof(float), cudaMemcpyDeviceToHost);

            sf = sqrtf(T0 / h_Temp);
            for (int k = 0; k < N; k++) { // reescaleo de velocidades
                h_vx[k] *= sf;
                h_vy[k] *= sf;
                h_vz[k] *= sf;
            }
            cudaDeviceSynchronize();

            if (i % TMES == 0) {
                // Copiar datos necesarios al host para loggeo
                float epot_host, pres_host, ekin_host;
                cudaMemcpy(&epot_host, d_Epot, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&pres_host, d_Pres, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&ekin_host, d_Ekin, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_rx, d_rx, N * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_ry, d_ry, N * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_rz, d_rz, N * sizeof(float), cudaMemcpyDeviceToHost);

                epot_host += Etail;
                pres_host += Ptail;

                epotm += epot_host;
                presm += pres_host;
                mes++;

                fprintf(file_thermo, "%f %f %f %f %f\n", t, temp_host, pres_host,
                        epot_host, epot_host + ekin_host);
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
