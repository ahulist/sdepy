__constant__ float m = 0.0f;
__constant__ float gamma = 0.0f;
__constant__ float a = 0.0f;
__constant__ float omega = 0.0f;
__constant__ float f = 0.0f;
__constant__ float D = 0.0f;
__constant__ float xi = 0.0f;
__constant__ float t = 0.0f;
__constant__ float dt = 0.0f;

__constant__ float steps = 10 d;

__device__ float U(float x)
{
	return 0;
}

__global__ void sim(float *out)
{
	float t = 0.0f;
	float x = 0.0f;
	float x_d = 0.0f;
	float x_dd = 0.0f;

	float x_dd = 1 / m * (-gamma * x_d - U(x) + a * cos(omega * t) + f + D * xi);

	for (int i = 0; i < steps; i++) {
		float x_d_next = x_d + x_dd * dt;
		float x_next = x + x_d * dt;
		float x_dd_next = 1 / m * (-gamma * x_d_next - U(x_next) + a * cos(omega * t) + f + D * xi);

		t += dt;

		x = x_next;
		x_d = x_d_next;
		x_dd = x_dd_next;
	}

	int idx = threadIdx.x + threadIdx.y * 4;
	out[idx] = x;
}
