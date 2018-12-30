#define _USE_MATH_DEFINES
#include <math.h>
#include <curand_kernel.h>

__device__ curandState_t *curand_states[8];

__constant__ int steps_per_kernel_call = 1000;
__constant__ int steps_per_period = 800;
__constant__ int periods = 1000;
__constant__ int afterstep_every = 1000;

__constant__ float dt = 0.0015707931852085263;

__device__ double data[8][8];

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
	return -3.18412015595821 * v + 16.3186157992858 * cosf(5.00001 * t) -
	    6.36824031191641 * M_PI * cosf(2 * M_PI * x) + 3.18412015595821;
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
	float v = data[idx][2] = 0.0;	// v
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
		float newMean_v = avg_period_v + (v - avg_period_v) / (current_step + 1.0);
		avg_periods_v = avg_periods_v + (v - avg_period_v) * (v - newMean_v);
		avg_period_v = newMean_v;
		float newMean_x = avg_period_x + (x - avg_period_x) / (current_step + 1.0);
		avg_periods_x = avg_periods_x + (x - avg_period_x) * (x - newMean_x);
		avg_period_x = newMean_x;

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
		if (current_step % 1000 == 0) {
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
