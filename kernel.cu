#define _USE_MATH_DEFINES
#include <math.h>
#include <curand_kernel.h>

const int threads_total = 4;
__device__ curandState_t* curand_states[threads_total];

__constant__ int steps_per_kernel_call = 200;
__constant__ int steps_per_period = 2000;
__constant__ int periods = 1;
__constant__ int number_of_threads = 4;
__constant__ int afterstep_every = 1;

__constant__ float dt = 0.0020949113096826;

__shared__ float data[4][12];

__global__ void initkernel(int seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  curandState_t* s = new curandState_t;
  curand_init(seed, idx, 0, s);
  curand_states[idx] = s;
}

__device__ void calc_avg(float& current_avg, float new_value,
                         int current_step) {
  current_avg +=
      (new_value - current_avg) / (current_step % steps_per_period + 1);
}

extern "C" __global__ void prepare_simulation(float* summary, float* output) {
  int idx = threadIdx.x + threadIdx.y * 2;

  data[idx][0] = 0.0f;  // current step

  float position = data[idx][1] = 0.0;  // position
  data[idx][2] = 0.0f;                  // avg_period_position
  data[idx][3] = 0.0f;                  // avg_periods_position
  float velocity = data[idx][4] = 1.0;  // velocity
  data[idx][5] = 0.0f;                  // avg_period_velocity
  data[idx][6] = 0.0f;                  // avg_periods_velocity
  float t = data[idx][7] = 0.0;         // t

  data[idx][8] =
      -0.100694794079146 * velocity +
      0.201389588158292 * M_PI * sinf(2 * M_PI * position) +
      0.415366025576478 * cosf(3.749 * t) + 0.100694794079146;  // rhs

}

__device__ void afterstep(float position, float velocity, float rhs) {}

extern "C" __global__ void continue_simulation(float* summary, float* output) {
  int idx = threadIdx.x + threadIdx.y * 2;

  int current_step = (int) data[idx][0];
  float position = data[idx][1];
  float position_next;
  float avg_period_position = data[idx][2];
  float avg_periods_position = data[idx][3];
  float velocity = data[idx][4];
  float velocity_next;
  float avg_period_velocity = data[idx][5];
  float avg_periods_velocity = data[idx][6];
  float t = data[idx][7];
  float rhs = data[idx][8];
  float rhs_next;

  for (int i = 0; i < steps_per_kernel_call; i++) {
    // iterative mean https://stackoverflow.com/a/1934266/1185254
    calc_avg(avg_period_position, position, current_step);
    calc_avg(avg_period_velocity, velocity, current_step);

    if (current_step % (steps_per_period - 1) == 0) {
      calc_avg(avg_periods_position, avg_period_position,
               current_step / steps_per_period);
      avg_period_position = 0.0f;
      calc_avg(avg_periods_velocity, avg_period_velocity,
               current_step / steps_per_period);
      avg_period_velocity = 0.0f;
    }

    // rand_val = curand_uniform(curand_states[idx]);

    position_next = position + velocity * dt;
    velocity_next = velocity + rhs * dt;
    rhs_next = -0.100694794079146 * velocity +
               0.201389588158292 * M_PI * sinf(2 * M_PI * position) +
               0.415366025576478 * cosf(3.749 * t) + 0.100694794079146;

    t += dt;

    position = position_next;
    velocity = velocity_next;
    rhs = rhs_next;

    if (current_step % 1 == 0) {
      afterstep(position_next, velocity_next, rhs_next);
    }

    current_step += 1;
  }

  data[idx][0] = current_step;
  data[idx][1] = position;
  data[idx][2] = avg_period_position;
  data[idx][3] = avg_periods_position;
  data[idx][4] = velocity;
  data[idx][5] = avg_period_velocity;
  data[idx][6] = avg_periods_velocity;
  data[idx][7] = t;
  data[idx][8] = rhs;
}

extern "C" __global__ void end_simulation(float* summary, float* output) {
  int idx = threadIdx.x + threadIdx.y * 2;

  summary[idx * 6 + 0] = data[idx][1];  // position
  summary[idx * 6 + 1] = data[idx][2];  // avg_period_position
  summary[idx * 6 + 2] = data[idx][3];  // avg_periods_position
  summary[idx * 6 + 3] = data[idx][4];  // velocity
  summary[idx * 6 + 4] = data[idx][5];  // avg_period_velocity
  summary[idx * 6 + 5] = data[idx][6];  // avg_periods_velocity
}