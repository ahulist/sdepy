#define _USE_MATH_DEFINES
#include <math.h>
#include <curand_kernel.h>

const int threads_total = 4;
__device__ curandState_t *curand_states[threads_total];

__constant__ int steps_per_kernel_call = 200;
__constant__ int steps_per_period = 2000;
__constant__ int periods = 1;
__constant__ int number_of_threads = 4;
__constant__ int afterstep_every = 1;

__constant__ float dt = 0.0020949113096826;

__shared__ float data[4][12];

__global__ void initkernel(int seed)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curandState_t *s = new curandState_t;
	curand_init(seed, idx, 0, s);
	curand_states[idx] = s;
}

__device__ __inline__ float v_diff(float t, float v, float x)
{
	return -0.100694794079146 * v + 0.201389588158292 * M_PI * sinf(2 * M_PI * x) +
	    0.415366025576478 * cosf(3.749 * t) + 0.100694794079146;
}

__device__ __inline__ float x_diff(float t, float v, float x)
{
	return v;
}

__device__ __inline__ void calc_avg(float &current_avg, float new_value, int current_step)
{
	current_avg += (new_value - current_avg) / (current_step % steps_per_period + 1);
}

extern "C" __global__ void prepare_simulation(float *summary, float *output)
{
	int idx = threadIdx.x + threadIdx.y * 2;

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

extern "C" __global__ void continue_simulation(float *summary, float *output)
{
	int idx = threadIdx.x + threadIdx.y * 2;

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

	float rk4_v_diff_1 = v_diff(t, v, x);
	float rk4_x_diff_1 = x_diff(t, v, x);
	float rk4_v_diff_2 = v_diff(t + dt / 2.0, v + dt * rk4_v_diff_1 / 2.0, x + dt * rk4_x_diff_1 / 2.0);
	float rk4_x_diff_2 = x_diff(t + dt / 2.0, v + dt * rk4_v_diff_1 / 2.0, x + dt * rk4_x_diff_1 / 2.0);
	float rk4_v_diff_3 = v_diff(t + dt / 2.0, v + dt * rk4_v_diff_2 / 2.0, x + dt * rk4_x_diff_2 / 2.0);
	float rk4_x_diff_3 = x_diff(t + dt / 2.0, v + dt * rk4_v_diff_2 / 2.0, x + dt * rk4_x_diff_2 / 2.0);
	float rk4_v_diff_4 = v_diff(t + dt, v + dt * rk4_v_diff_3, x + dt * rk4_x_diff_3);
	float rk4_x_diff_4 = x_diff(t + dt, v + dt * rk4_v_diff_3, x + dt * rk4_x_diff_3);
	float rk4_v_diff = (rk4_v_diff_1 + 2 * rk4_v_diff_2 + 2 * rk4_v_diff_3 + rk4_v_diff_4) / 6.0;
	float rk4_x_diff = (rk4_x_diff_1 + 2 * rk4_x_diff_2 + 2 * rk4_x_diff_3 + rk4_x_diff_4) / 6.0;

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

		v_diff_value = rk4_v_diff;
		x_diff_value = rk4_x_diff;

		v_next = v + v_diff_value * dt;
		x_next = x + x_diff_value * dt;

		t += dt;

		v = v_next;
		x = x_next;

		rk4_v_diff_1 = v_diff(t, v, x);
		rk4_x_diff_1 = x_diff(t, v, x);
		rk4_v_diff_2 = v_diff(t + dt / 2.0, v + dt * rk4_v_diff_1 / 2.0, x + dt * rk4_x_diff_1 / 2.0);
		rk4_x_diff_2 = x_diff(t + dt / 2.0, v + dt * rk4_v_diff_1 / 2.0, x + dt * rk4_x_diff_1 / 2.0);
		rk4_v_diff_3 = v_diff(t + dt / 2.0, v + dt * rk4_v_diff_2 / 2.0, x + dt * rk4_x_diff_2 / 2.0);
		rk4_x_diff_3 = x_diff(t + dt / 2.0, v + dt * rk4_v_diff_2 / 2.0, x + dt * rk4_x_diff_2 / 2.0);
		rk4_v_diff_4 = v_diff(t + dt, v + dt * rk4_v_diff_3, x + dt * rk4_x_diff_3);
		rk4_x_diff_4 = x_diff(t + dt, v + dt * rk4_v_diff_3, x + dt * rk4_x_diff_3);
		rk4_v_diff = (rk4_v_diff_1 + 2 * rk4_v_diff_2 + 2 * rk4_v_diff_3 + rk4_v_diff_4) / 6.0;
		rk4_x_diff = (rk4_x_diff_1 + 2 * rk4_x_diff_2 + 2 * rk4_x_diff_3 + rk4_x_diff_4) / 6.0;

	/**
    	 * Afterstep
    	 */
		if (current_step % 1 == 0) {
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

extern "C" __global__ void end_simulation(float *summary, float *output)
{
	int idx = threadIdx.x + threadIdx.y * 2;

	summary[idx * 6 + 0] = data[idx][2];	// v
	summary[idx * 6 + 1] = data[idx][3];	// avg_period_v
	summary[idx * 6 + 2] = data[idx][4];	// avg_periods_v
	summary[idx * 6 + 3] = data[idx][5];	// x
	summary[idx * 6 + 4] = data[idx][6];	// avg_period_x
	summary[idx * 6 + 5] = data[idx][7];	// avg_periods_x
}
