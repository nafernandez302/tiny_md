#ifndef CORE_H
#define CORE_H

#ifdef __cplusplus
extern "C" {
#endif

void init_pos(float* rx, float* ry, float* rz, const float rho);
void init_vel(float* vx, float* vy, float* vz, float* temp, float* ekin);

// Aquí cambiamos temp a const float*
void forces(const float* rx, const float* ry, const float* rz,
            float* fx, float* fy, float* fz,
            float* epot, float* pres,
            const float* temp,     // ← ahora const
            float rho,
            float V, float L);

// Declaramos el wrapper CUDA con la misma firma
void forces_cu(const float* rx, const float* ry, const float* rz,
               float* fx, float* fy, float* fz,
               float* epot, float* pres,
               const float* temp,    // ← igual aquí
               float rho,
               float V, float L);

void velocity_verlet(float* rx, float* ry, float* rz,
                     float* vx, float* vy, float* vz,
                     float* fx, float* fy, float* fz,
                     float* epot, float* ekin,
                     float* pres, float* temp,
                     const float rho,
                     const float V, const float L);

#ifdef __cplusplus
}
#endif

#endif // CORE_H
