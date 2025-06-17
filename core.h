#ifndef CORE_H
#define CORE_H

__host__ void init_pos(float* rx, float* ry, float* rz, const float rho);
__host__ void init_vel(float* vx, float* vy, float* vz, float* temp, float* ekin);
__global__ void forces(const float* rx, const float* ry, const float* rz,
                       float* fx, float* fy, float* fz, float* epot, float* pres,
                       float* temp, float rho, float V, float L);
void velocity_verlet(float* rx, float* ry, float* rz, float* vx, float* vy, float* vz,
                     float* fx, float* fy, float* fz, float* epot,
                     float* ekin, float* pres, float* temp, const float rho,
                     const float V, const float L);

#endif
