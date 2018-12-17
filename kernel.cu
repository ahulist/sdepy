#define _USE_MATH_DEFINES
#include <math.h>
#include <curand_kernel.h>

__device__ curandState_t *curand_states[15360];

__constant__ int steps_per_kernel_call = 200;
__constant__ int steps_per_period = 2000;
__constant__ int periods = 2;
__constant__ int afterstep_every = 10;

__constant__ float dt = 0.0020949113096826;

__device__ float parameter_D_values[2] = { 0.2, 2.0 };
__device__ float parameter_f_values[1] = { 1.0 };
__device__ float parameter_omega_values[1] = { 3.749 };
__device__ float parameter_a_values[5] = { 4.125, 4.625, 5.125, 5.625, 6.125 };
__device__ float parameter_gamma_values[1] = { 1.0 };
__device__ float parameter_m_values[3] = { 9.931, 9.831, 9.731 };

__device__ int parameters_values_lens[6] = {
	2, 1, 1, 5, 1, 3
};

__device__ float *parameters_values[6] = {
	parameter_D_values, parameter_f_values, parameter_omega_values, parameter_a_values, parameter_gamma_values,
	    parameter_m_values
};

__device__ float data[15360][8];

__global__ void initkernel(int seed)
{
	//int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int idx =
	    blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) +
	    (threadIdx.y * blockDim.x) + threadIdx.x;

	curandState_t *s = new curandState_t;
	curand_init(seed * idx, idx, 0, s);	// czy mnozenie przez idx jest potrzebne?
	curand_states[idx] = s;
}

__device__ __inline__ float v_diff(int idx, float t, float v, float x)
{
	return -0.100694794079146 * v + 0.0201389588158292 * curand_uniform(curand_states[idx]) +
	    0.201389588158292 * M_PI * sinf(2 * M_PI * x) + 0.415366025576478 * cosf(3.749 * t) + 0.100694794079146;
}

__device__ __inline__ float x_diff(int idx, float t, float v, float x)
{
	return v;
}

__device__ __inline__ void calc_avg(float &current_avg, float new_value, int current_step)
{
	current_avg += (new_value - current_avg) / (current_step % steps_per_period + 1);
}

extern "C" __global__ void prepare_simulation()
{
	//int idx =  blockIdx.x  *blockDim.x + threadIdx.x;
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int idx =
	    blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) +
	    (threadIdx.y * blockDim.x) + threadIdx.x;

	data[idx][0] = 0.0f;	// current step

	float t = data[idx][1] = 0.0;	// t
	float v = data[idx][2] = 1.0;	// v
	data[idx][3] = 0.0f;	// avg_period_v
	data[idx][4] = 0.0f;	// avg_periods_v
	float x = data[idx][5] = 0.0;	// x
	data[idx][6] = 0.0f;	// avg_period_x
	data[idx][7] = 0.0f;	// avg_periods_x
}

__device__ void afterstep(float t, float v, float x)
{

}

extern "C" __global__ void continue_simulation()
{
	//int idx =  blockIdx.x * blockDim.x + threadIdx.x;
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int idx =
	    blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) +
	    (threadIdx.y * blockDim.x) + threadIdx.x;

	float my_params[6];
	int index = (int)idx / 500;
	for (int i = 0; i < 6; i++) {
		my_params[i] = parameters_values[i][index % parameters_values_lens[i]];
		index = (int)index / parameters_values_lens[i];
	}

	int current_step = (int)data[idx][0];
	float t = data[idx][1];
	float v = data[idx][2];
	float v_next;
	float v_diff_value;
	float avg_period_v = data[idx][3];
	float avg_periods_v = data[idx][4];
	float x = data[idx][5];
	float x_next;
	float x_diff_value;
	float avg_period_x = data[idx][6];
	float avg_periods_x = data[idx][7];

	float rk4_v_diff_1 = v_diff(idx, t, v, x);
	float rk4_x_diff_1 = x_diff(idx, t, v, x);
	float rk4_v_diff_2 = v_diff(idx, t + dt / 2.0, v + dt * rk4_v_diff_1 / 2.0, x + dt * rk4_x_diff_1 / 2.0);
	float rk4_x_diff_2 = x_diff(idx, t + dt / 2.0, v + dt * rk4_v_diff_1 / 2.0, x + dt * rk4_x_diff_1 / 2.0);
	float rk4_v_diff_3 = v_diff(idx, t + dt / 2.0, v + dt * rk4_v_diff_2 / 2.0, x + dt * rk4_x_diff_2 / 2.0);
	float rk4_x_diff_3 = x_diff(idx, t + dt / 2.0, v + dt * rk4_v_diff_2 / 2.0, x + dt * rk4_x_diff_2 / 2.0);
	float rk4_v_diff_4 = v_diff(idx, t + dt, v + dt * rk4_v_diff_3, x + dt * rk4_x_diff_3);
	float rk4_x_diff_4 = x_diff(idx, t + dt, v + dt * rk4_v_diff_3, x + dt * rk4_x_diff_3);
	v_diff_value = (rk4_v_diff_1 + 2 * rk4_v_diff_2 + 2 * rk4_v_diff_3 + rk4_v_diff_4) / 6.0;
	x_diff_value = (rk4_x_diff_1 + 2 * rk4_x_diff_2 + 2 * rk4_x_diff_3 + rk4_x_diff_4) / 6.0;

	for (int i = 0; i < steps_per_kernel_call; i++) {
	/**
         * Averaging
         */
		// iterative mean https://stackoverflow.com/a/1934266/1185254
		calc_avg(avg_period_v, v, current_step);
		calc_avg(avg_period_x, x, current_step);

		if (current_step % (steps_per_period - 1) == 0) {
			calc_avg(avg_periods_v, avg_period_v, current_step / steps_per_period);
			avg_period_v = 0.0f;
			calc_avg(avg_periods_x, avg_period_x, current_step / steps_per_period);
			avg_period_x = 0.0f;
		}

	/**
    	 * Integration
    	 */
		v_next = v + v_diff_value * dt;
		x_next = x + x_diff_value * dt;

		t += dt;

		v = v_next;
		x = x_next;

		rk4_v_diff_1 = v_diff(idx, t, v, x);
		rk4_x_diff_1 = x_diff(idx, t, v, x);
		rk4_v_diff_2 = v_diff(idx, t + dt / 2.0, v + dt * rk4_v_diff_1 / 2.0, x + dt * rk4_x_diff_1 / 2.0);
		rk4_x_diff_2 = x_diff(idx, t + dt / 2.0, v + dt * rk4_v_diff_1 / 2.0, x + dt * rk4_x_diff_1 / 2.0);
		rk4_v_diff_3 = v_diff(idx, t + dt / 2.0, v + dt * rk4_v_diff_2 / 2.0, x + dt * rk4_x_diff_2 / 2.0);
		rk4_x_diff_3 = x_diff(idx, t + dt / 2.0, v + dt * rk4_v_diff_2 / 2.0, x + dt * rk4_x_diff_2 / 2.0);
		rk4_v_diff_4 = v_diff(idx, t + dt, v + dt * rk4_v_diff_3, x + dt * rk4_x_diff_3);
		rk4_x_diff_4 = x_diff(idx, t + dt, v + dt * rk4_v_diff_3, x + dt * rk4_x_diff_3);
		v_diff_value = (rk4_v_diff_1 + 2 * rk4_v_diff_2 + 2 * rk4_v_diff_3 + rk4_v_diff_4) / 6.0;
		x_diff_value = (rk4_x_diff_1 + 2 * rk4_x_diff_2 + 2 * rk4_x_diff_3 + rk4_x_diff_4) / 6.0;

	/**
    	 * Afterstep
    	 */
		if (current_step % 10 == 0) {
			afterstep(t, v, x);
		}

		current_step += 1;
	}

	data[idx][0] = current_step;
	data[idx][1] = t;
	data[idx][2] = v;
	data[idx][3] = avg_period_v;
	data[idx][4] = avg_periods_v;
	data[idx][5] = x;
	data[idx][6] = avg_period_x;
	data[idx][7] = avg_periods_x;
}

extern "C" __global__ void end_simulation(float *summary)
{
	//int idx =  blockIdx.x * blockDim.x + threadIdx.x;
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int idx =
	    blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) +
	    (threadIdx.y * blockDim.x) + threadIdx.x;

	summary[idx * 6 + 0] = data[idx][2];	// v
	summary[idx * 6 + 1] = data[idx][3];	// avg_period_v
	summary[idx * 6 + 2] = data[idx][4];	// avg_periods_v
	summary[idx * 6 + 3] = data[idx][5];	// x
	summary[idx * 6 + 4] = data[idx][6];	// avg_period_x
	summary[idx * 6 + 5] = data[idx][7];	// avg_periods_x
}
