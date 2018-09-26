#define _USE_MATH_DEFINES
#include <math.h>
#include <curand_kernel.h>

const int nstates = 4;
__device__ curandState_t* states[nstates];

__constant__ float m = 0.5095f;
__constant__ float gamma_ = 1.0f;
__constant__ float a = 24.9375f;
__constant__ float omega = 11.93f;
__constant__ float f = -9.81f;
__constant__ float D = 0.0f;
__constant__ float xi = 0.0f;
__constant__ float dt = 0.002095f; // 2*PI/(omega*steps_per_period)

__constant__ int periods = 1000;
__constant__ int steps_per_period = 800;
__constant__ int steps_per_kernel_call = 200;

__global__ void initkernel(int seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t* s = new curandState_t;
    curand_init(seed, idx, 0, s);
    states[idx] = s;

    //if (idx < nstates) {
    //    curandState_t* s = new curandState_t;
    //    if (s != 0) {
    //        curand_init(seed, idx, 0, s);
    //    }
    //    states[idx] = s;
    //}
}

// __global__ void randfillkernel(float *values, int N)
// {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     curandState_t s = *states[idx];
//     for(int i=idx; i < N; i += blockDim.x * gridDim.x) {
//         values[i] = curand_uniform(&s);
//     }
//     *states[idx] = s;

//     //if (idx < nstates) {
//     //    curandState_t s = *states[idx];
//     //    for(int i=idx; i < N; i += blockDim.x * gridDim.x) {
//     //        values[i] = curand_uniform(&s);
//     //    }
//     //    *states[idx] = s;
//     //}
// }

__device__ float U_d(float x)
{
	return 0.0;
}

__shared__ float data [4][12];
/*
0:  avg_over_period_position
1:  avg_over_period_velocity
2:  avg_over_periods_position
3:  avg_over_periods_velocity
4:  t
5:  position
6:  velocity
7:  acceleration
8:  velocity_next
9:  position_next
10: acceleration_next
11: current_step
*/

__device__ void calc_avg(float &current_avg, float new_value, float current_step)
{
    current_avg += (new_value - current_avg) / (current_step / steps_per_period + 1);
}

__global__ void prepare_simulation()
{
    int idx = threadIdx.x + threadIdx.y * 2;
    
    data[idx][0] = 0.0f; // avg_over_period_position
    data[idx][1] = 0.0f; // avg_over_period_velocity
    data[idx][2] = 0.0f; // avg_over_periods_position
    data[idx][3] = 0.0f; // avg_over_periods_velocity
    data[idx][4] = 0.0f; // t
    data[idx][5] = 0.0f; // position
    data[idx][6] = 1.0f; // velocity
    data[idx][7] = 1.0 / m * (-gamma_ * data[idx][6] - U_d(data[idx][5]) + a * cos(omega * data[idx][4]) + f + D * xi); // acceleration
    data[idx][11] = 0.0f;
}

__global__ void continue_simulation()
{
    int idx = threadIdx.x + threadIdx.y * 2;
    
    float avg_over_period_position = data[idx][0];
    float avg_over_period_velocity = data[idx][1];
    float avg_over_periods_position = data[idx][2];
    float avg_over_periods_velocity = data[idx][3];
    float t = data[idx][4];
    float position = data[idx][5];
    float velocity = data[idx][6];
    float acceleration = data[idx][7];
    float velocity_next;
    float position_next;
    float acceleration_next;
    int current_step = (int) data[idx][11];

    for (int i = 0; i < steps_per_kernel_call; i++) {
        velocity_next = velocity + acceleration * dt;
        position_next = position + velocity * dt;
        acceleration_next = 1 / m * (-gamma_ * velocity_next - U_d(position_next) + a * cos(omega * t) + f + D * xi);
        // rand_val = curand_uniform(states[idx]);

        t += dt;

        position = position_next;
        velocity = velocity_next;
        acceleration = acceleration_next;
        
        // iterative mean https://stackoverflow.com/a/1934266/1185254
        calc_avg(avg_over_period_position, position, current_step);
        calc_avg(avg_over_period_velocity, velocity, current_step);
        
        if(current_step == steps_per_period){
            calc_avg(avg_over_periods_position, avg_over_period_position, current_step);
            calc_avg(avg_over_periods_velocity, avg_over_period_velocity, current_step);
            avg_over_period_position = 0.0f;
            avg_over_period_velocity = 0.0f;
        }
        
        current_step += 1;
    }
    
    data[idx][0] = avg_over_period_position;
    data[idx][1] = avg_over_period_velocity;
    data[idx][2] = avg_over_periods_position;
    data[idx][3] = avg_over_periods_velocity;
    data[idx][4] = t;
    data[idx][5] = position;
    data[idx][6] = velocity;
    data[idx][7] = acceleration;
    data[idx][11] = current_step;
}

__global__ void end_simulation(float *out)
{
    int idx = threadIdx.x + threadIdx.y * 2;
    out[idx*6+0] = 0;//data[idx][0];
    out[idx*6+1] = 0;//data[idx][1];
    out[idx*6+2] = data[idx][2];
    out[idx*6+3] = data[idx][3];
    out[idx*6+4] = data[idx][5];
    out[idx*6+5] = data[idx][6];
}


/*__device__ float diffusion(float l_gam, float l_Dg, float l_dt, int l_2ndorder, curandState *l_state)
{
    if (l_Dg != 0.0f) {
        float r = curand_uniform(l_state);
        if (l_2ndorder) {
            if ( r <= 1.0f/6 ) {
                return -sqrtf(6.0f*l_gam*l_Dg*l_dt);
            } else if ( r > 1.0f/6 && r <= 2.0f/6 ) {
                return sqrtf(6.0f*l_gam*l_Dg*l_dt);
            } else {
                return 0.0f;
            }
        } else {
            if ( r <= 0.5f ) {
                return -sqrtf(2.0f*l_gam*l_Dg*l_dt);
            } else {
                return sqrtf(2.0f*l_gam*l_Dg*l_dt);
            }
        }
    } else {
        return 0.0f;
    }
}*/