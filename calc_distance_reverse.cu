#include <stdio.h>
#include <stdlib.h>

class v_point
{
	public:
	
	long position;
	int red;
	int blue;
	int green;
};

#define BLOCK_SIZE 24
 
__global__ void cu_calc_dist(v_point *pixels_d, v_point *centers_d, long array_size, long centers_size)
{
    long i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	long j = 0;
	long x_point, y_point, x_center, y_center;
	float distance, shortest_dist;
	v_point closest_center;
	
	if(i < array_size*array_size){
		x_point = pixels_d[i].position / array_size;
		y_point = pixels_d[i].position % array_size;
		
		for(j = 0; j < centers_size; j++){
			x_center = centers_d[j].position / array_size;
			y_center = centers_d[j].position % array_size;	
			distance = sqrt(pow(1.0 * x_center - x_point, 2) + pow(1.0 * y_center - y_point, 2));
			
			if(j == 0){	
				shortest_dist = distance;
				closest_center.position = centers_d[j].position;
				closest_center.red = centers_d[j].red;
				closest_center.blue = centers_d[j].blue;
				closest_center.green = centers_d[j].green;
			
			// if not, then check to see if the new distance we calculated is smaller
			// note this produces a first calculated point for contested areas
			} else if(distance > shortest_dist){
				shortest_dist = distance;
				closest_center.position = centers_d[j].position;
				closest_center.red = centers_d[j].red;
				closest_center.blue = centers_d[j].blue;
				closest_center.green = centers_d[j].green;
			} 
		}
		
		pixels_d[i] = closest_center;

	}
}

// This function is called from the host computer.
// It manages memory and calls the function that is executed on the GPU
extern void calc_distance(v_point *pixels, v_point *centers, long array_size, long centers_size)
{
	// build GPU counterpart for each class array on host
	v_point *pixels_d;
	v_point *centers_d;
	cudaError_t result;

	// allocate space in the device 
	result = cudaMalloc ((void**) &pixels_d, sizeof(v_point) * array_size * array_size);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMalloc - 'pixels' failed.");
		exit(1);		
	}
	
	result = cudaMalloc ((void**) &centers_d, sizeof(v_point) * centers_size);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMalloc - 'centers' failed.");
		exit(1);
	}

	//copy the array from host to *_d in the device 
	result = cudaMemcpy (pixels_d, pixels, sizeof(v_point) * array_size * array_size, cudaMemcpyHostToDevice);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy - 'pixels' failed.");
		exit(1);
	}
	
	result = cudaMemcpy (centers_d, centers, sizeof(v_point) * centers_size, cudaMemcpyHostToDevice);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy - 'centers' failed.");
		exit(1);
	}

	// set execution configuration
	dim3 dimblock (BLOCK_SIZE);
	dim3 dimgrid (ceil((float) array_size*array_size/BLOCK_SIZE));

	// actual computation: Call the kernel
	cu_calc_dist <<<dimgrid, dimblock>>> (pixels_d, centers_d, array_size, centers_size);

	// transfer results back to host
	result = cudaMemcpy (pixels, pixels_d, sizeof(v_point) * array_size * array_size, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy copy to host failed. %s\n", cudaGetErrorString(result));
		exit(1);
		cudaFree(pixels_d);
		cudaFree(centers_d);
	}
	
	// release the memory on the GPU 
	cudaFree(pixels_d);
	cudaFree(centers_d);
}