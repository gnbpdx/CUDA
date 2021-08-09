#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <string>
#include <iostream>
#define BLOCK_DIM 1024
using namespace cv;
__device__ void three_by_three_multiplication(float* matrix1, float* matrix2, float* result) {
	float num = 0;
	for (int row = 0; row < 3; ++row) {
		for (int col = 0; col < 3; ++col) {
			for (int i = 0; i < 3; ++i) {
				num += matrix1[3 * row + i]	* matrix2[3 * i + col];
			}
			result[3 * row + col] = num;
			num = 0;
		}
	
	}
}
__device__ void matrix_vector(float* matrix, float* vector, float* result) {
	float num = 0;
	for (int i = 0; i < 3; ++i) {
		for (int index = 0; index < 3; ++index) {
			num += matrix[3 * i + index] * vector[index];
		}
		result[i] = num;
		num = 0;
	}
}
__global__ void convert_color_space_BGR_to_RGB(float* img_BGR, float* img_RGB, int size) {
	int start = (blockDim.x * blockIdx.x + threadIdx.x) * 3;
	if (start < size) {
		img_RGB[start] = img_BGR[start + 2];
		img_RGB[start + 1] = img_BGR[start + 1];
		img_RGB[start + 2] = img_BGR[start];
	}

}
__global__ void convert_color_space_RGB_to_BGR(float* img_RGB, float* img_BGR, int size) {
	int start = (blockDim.x * blockIdx.x + threadIdx.x) * 3;
	if (start < size) {
		img_BGR[start] = img_RGB[start + 2];
		img_BGR[start + 1] = img_RGB[start + 1];
		img_BGR[start + 2] = img_RGB[start];
	}
}
__global__ void convert_color_space_RGB_to_Lab(float* img_RGB, float* img_Lab, float* temp, int size, float* RGB_to_Lab1, float* RGB_to_Lab2, float* RGB_to_Lab3) {
	int start = (blockDim.x * blockIdx.x + threadIdx.x) * 3;
	if (start < size) {
		float array[3], array2[3], array3[3];
		for (int i = 0; i < 3; ++i) {
			array[i] = img_RGB[start + i];
		}
		matrix_vector(RGB_to_Lab1, array, array2);
		three_by_three_multiplication(RGB_to_Lab2, RGB_to_Lab3, temp);	
		for (int i = 0; i < 3; ++i) {
			if (array2[i] == 0)
				array2[i] = 0.0001; //Makes log function well-defined
			array2[i] = log10f(array2[i]);
		}
		matrix_vector(temp, array2, array3);
		for (int i = 0; i < 3; ++i) {
			img_Lab[start + i] = array3[i];
		}			
	}
}
__global__ void convert_color_space_Lab_to_RGB(float* img_Lab, float* img_RGB, float* temp, int size, float* Lab_to_RGB1, float* Lab_to_RGB2, float* Lab_to_RGB3) {
	int start = (blockDim.x * blockIdx.x + threadIdx.x) * 3;
	if (start < size) {
		float array[3], array2[3], array3[3];
		for (int i = 0; i < 3; ++i) {
			array[i] = img_Lab[start + i];
		}		
		three_by_three_multiplication(Lab_to_RGB1, Lab_to_RGB2, temp);
		matrix_vector(temp, array, array2);
		for (int i = 0; i < 3; ++i) {
			array2[i] = powf(10, array2[i]);
		}
		matrix_vector(Lab_to_RGB3, array2, array3);
		for (int i = 0; i < 3; ++i) {
			img_RGB[start + i] = array3[i];
		}
	}
}
__global__ void convert_color_space_RGB_to_CIECAM97s(float* img_RGB, float* img_CIECAM97s, float* temp, int size, float* RGB_to_CIECAM97s1, float* RGB_to_CIECAM97s2) {
	int start = (blockDim.x * blockIdx.x + threadIdx.x) * 3;
	if (start < size) {
		float array[3], array2[3], array3[3];
		for (int i = 0; i < 3; ++i) {
			array[i] = img_RGB[start + i];	
		}
		matrix_vector(RGB_to_CIECAM97s1, array, array2);
		matrix_vector(RGB_to_CIECAM97s2, array2, array3);
		for (int i = 0; i < 3; ++i) {
			img_CIECAM97s[start + i] = array3[i];	
		}
	}
}
__global__ void convert_color_space_CIECAM97s_to_RGB(float* img_CIECAM97s, float* img_RGB, float* temp, int size, float* CIECAM97s_to_RGB1, float* CIECAM97s_to_RGB2) {	
	int start = (blockDim.x * blockIdx.x + threadIdx.x) * 3;
	if (start < size) {		
		float array[3], array2[3], array3[3];
		for (int i = 0; i < 3; ++i) {
			array[i] = img_CIECAM97s[start + i];	
		}
		matrix_vector(CIECAM97s_to_RGB1, array, array2);
		matrix_vector(CIECAM97s_to_RGB2, array2, array3);
		for (int i = 0; i < 3; ++i) {
			img_RGB[start + i] = array3[i];
		}
	}
}
__global__ void mean_add(float* src, float* sum, int size) {	
	int start = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
	if (start < size) {	
		for (int i = 0; i < 3; ++i) {
			atomicAdd(sum + i, src[start + i]);	
		}
	}
}
__global__ void std_add(float* src, float* mean, float* sum, int size) {
	int start = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
	if (start < size) {
		for (int i = 0; i < 3; ++i) {
			atomicAdd(sum + i, powf(src[start + i] - mean[i], 2));
		}	
	}
}
__global__ void calculate_mean(float* src, float* mean, int size) {
	int thread = threadIdx.x;
	int num_blocks = ceil(size/(float)BLOCK_DIM);
	if (threadIdx.x == 0) {
		mean_add<<<num_blocks, BLOCK_DIM>>>(src, mean, size);		
	}
	cudaDeviceSynchronize();
	mean[thread] = mean[thread] / size;	
}
__global__ void calculate_standard_deviation(float* src, float* mean, float* sigma, int size) {
	int thread = threadIdx.x;
	int num_blocks = ceil(size/(float)BLOCK_DIM);
	if (threadIdx.x == 0) {
		std_add<<<num_blocks, BLOCK_DIM>>>(src, mean, sigma, size);
	}
	cudaDeviceSynchronize();
	sigma[thread] = sqrtf(sigma[thread] / size);
	
}
__global__ void transfer_image(float* source, float* target, float* result, float* mean_source, float* mean_target, float* sigma_source, float* sigma_target, int size) {
	int start = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
	if (start < size) {
		for (int i = 0; i < 3; ++i) {
			result[start + i] = (source[start + i] - mean_source[i]) * (sigma_target[i] / sigma_source[i]) + mean_target[i];
		}	
	}
}
void color_transfer(float* RGB_source, float* RGB_target, float* img_RGB_result, float* temp, std::string option, int size_source, int size_target) {
//Load transformation matrices
	float M1[] = {
    		0.3811, 0.5783, 0.0402,
    		0.1967, 0.7244, 0.0782,
    		0.0241, 0.1288, 0.8444
  	};
	float M2[] = {
		(float)(1/std::sqrt(3)), 0, 0,
    		0, (float)(1/std::sqrt(6)), 0,
    		0, 0, (float)(1/std::sqrt(2))
  	};
	float M3[] = {
    		1, 1, 1,
    		1, 1, -2,
    		1, -1, 0
  	};
	float M4[] = {
		1, 1, 1,
		1, 1, -1,
		1, -2, 0
	};
	float M5[] = {
		(float)(std::sqrt(3)/3), 0, 0,
		0, (float)(std::sqrt(6)/6), 0,
		0, 0, (float)(std::sqrt(2)/2)
	};
	float M6[] = {
		4.4679, -3.5873, 0.1193,
		-1.2186, 2.3809, -0.1624,
		0.0497, -0.2439, 1.2045
	};
	float M7[] = {
		0.3811, 0.5783, 0.0402,
		0.1967, 0.7244, 0.0782,
		0.0241, 0.1288, 0.8444
	};
	float M8[] = {
		2.00, 1.00, 0.05,
		1.00, -1.09, 0.09,
		0.11, 0.11, -0.22
	};
	float M9[]  {
		0.327869, 0.321594, 0.20077,
		0.327869, -0.635344, -0.185398,
		0.327869, -0.156875, -4.53512
	};
	float M10[] = {
		4.4679, -3.5873, 0.1193,
		-1.2186, 2.3809, -0.1624,
		0.0497, -0.2439, 1.2045
	};
	float *RGB_to_Lab1, *RGB_to_Lab2, *RGB_to_Lab3, *Lab_to_RGB1, *Lab_to_RGB2, *Lab_to_RGB3;
	float *RGB_to_CIECAM97s1, *RGB_to_CIECAM97s2, *CIECAM97s_to_RGB1, *CIECAM97s_to_RGB2;
	cudaMalloc(&RGB_to_Lab1, 9 * sizeof(float));
	cudaMalloc(&RGB_to_Lab2, 9 * sizeof(float));
	cudaMalloc(&RGB_to_Lab3, 9 * sizeof(float));
	cudaMalloc(&Lab_to_RGB1, 9 * sizeof(float));
	cudaMalloc(&Lab_to_RGB2, 9 * sizeof(float));
	cudaMalloc(&Lab_to_RGB3, 9 * sizeof(float));
	cudaMalloc(&RGB_to_CIECAM97s1, 9 * sizeof(float));
	cudaMalloc(&RGB_to_CIECAM97s2, 9 * sizeof(float));
	cudaMalloc(&CIECAM97s_to_RGB1, 9 * sizeof(float));
	cudaMalloc(&CIECAM97s_to_RGB2, 9 * sizeof(float));
	cudaMemcpy(RGB_to_Lab1, M1, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(RGB_to_Lab2, M2, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(RGB_to_Lab3, M3, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Lab_to_RGB1, M4, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Lab_to_RGB2, M5, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Lab_to_RGB3, M6, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(RGB_to_CIECAM97s1, M7, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(RGB_to_CIECAM97s2, M8, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(CIECAM97s_to_RGB1, M9, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(CIECAM97s_to_RGB2, M10, 9 * sizeof(float), cudaMemcpyHostToDevice);
	float* mean_source, *mean_target; //mean of source and target images
	float* sigma_source, *sigma_target; //standard deviation of source and target images
	float* img_source, *img_target, *img_result;	
	int num_source_blocks = ceil(size_source/((float)BLOCK_DIM));
	int num_target_blocks = ceil(size_target/((float)BLOCK_DIM));
	float zero_array[3] = {0};
//Allocate memory for mean and standard deviation of source and target images
	cudaMalloc(&mean_source, sizeof(float) * 3);
	cudaMalloc(&mean_target, sizeof(float) * 3);
	cudaMalloc(&sigma_source, sizeof(float) * 3);
	cudaMalloc(&sigma_target, sizeof(float) * 3);
	cudaMemcpy(mean_source, zero_array, sizeof(float) * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(mean_target, zero_array, sizeof(float) * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(sigma_source, zero_array, sizeof(float) * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(sigma_target, zero_array, sizeof(float) * 3, cudaMemcpyHostToDevice);
//Transfer source and target to appropriate color space
	if (option == "in_RGB") {
		img_source = RGB_source;
		img_target = RGB_target;
		img_result = img_RGB_result;
	}
	else if (option == "in_Lab") {
		cudaMalloc(&img_source, sizeof(float) * size_source);
		cudaMalloc(&img_target, sizeof(float) * size_target);
		cudaMalloc(&img_result, sizeof(float) * size_source);
		convert_color_space_RGB_to_Lab<<<num_source_blocks, BLOCK_DIM>>>(RGB_source, img_source, temp, size_source, RGB_to_Lab1, RGB_to_Lab2, RGB_to_Lab3);
		convert_color_space_RGB_to_Lab<<<num_target_blocks, BLOCK_DIM>>>(RGB_target, img_target, temp, size_target, RGB_to_Lab1, RGB_to_Lab2, RGB_to_Lab3);
	}
	else if (option == "in_CIECAM97s") {
		cudaMalloc(&img_source, sizeof(float) * size_source);
		cudaMalloc(&img_target, sizeof(float) * size_target);		
		cudaMalloc(&img_result, sizeof(float) * size_source);
		convert_color_space_RGB_to_CIECAM97s<<<num_source_blocks, BLOCK_DIM>>>(RGB_source, img_source, temp, size_source, RGB_to_CIECAM97s1, RGB_to_CIECAM97s2);
		convert_color_space_RGB_to_CIECAM97s<<<num_target_blocks, BLOCK_DIM>>>(RGB_target, img_target, temp, size_target, RGB_to_CIECAM97s1, RGB_to_CIECAM97s2);
	}
//Calculate mean of source and target
	calculate_mean<<<1, 3>>>(img_source, mean_source, size_source);
	calculate_mean<<<1, 3>>>(img_target, mean_target, size_target);
//Calculate standard deviation of source and target
	calculate_standard_deviation<<<1, 3>>>(img_source, mean_source, sigma_source, size_source);
	calculate_standard_deviation<<<1, 3>>>(img_target, mean_target, sigma_target, size_target);
//Transfer image
	transfer_image<<<num_source_blocks, BLOCK_DIM>>>(img_source, img_target, img_result, mean_source, mean_target, sigma_source, sigma_target, size_source);	
//Transfer result back to RGB
	if (option == "in_Lab") {
		convert_color_space_Lab_to_RGB<<<num_source_blocks, BLOCK_DIM>>>(img_result, img_RGB_result, temp, size_source, Lab_to_RGB1, Lab_to_RGB2, Lab_to_RGB3);
		cudaFree(&img_source);
		cudaFree(&img_target);
		cudaFree(&img_result);
	}
	else if (option == "in_CIECAM97s") {	
		convert_color_space_CIECAM97s_to_RGB<<<num_source_blocks, BLOCK_DIM>>>(img_result, img_RGB_result, temp, size_source, CIECAM97s_to_RGB1, CIECAM97s_to_RGB2);
		cudaFree(&img_source);
		cudaFree(&img_target);
		cudaFree(&img_result);
	}	
//Deallocate memory
	cudaFree(&mean_source);
	cudaFree(&mean_target);
	cudaFree(&sigma_source);
	cudaFree(&sigma_target);
}
int main(int argc, char** argv) {
	float* temp_array;
	cudaMalloc(&temp_array, 9 * sizeof(float));
//Read input images
	Mat img_BGR_source = imread(argv[1], 1);
	img_BGR_source.convertTo(img_BGR_source, CV_32FC3);
	Mat img_BGR_target = imread(argv[2], 1);
	img_BGR_target.convertTo(img_BGR_target, CV_32FC3);
//Allocate Space on Device
	int source_rows = img_BGR_source.rows;
	int source_cols = img_BGR_source.cols;
	int target_rows = img_BGR_target.rows;
	int target_cols = img_BGR_target.cols;
	float* d_BGR_source, *d_RGB_source, *d_BGR_target, *d_RGB_target;
	float* d_Lab_source, *d_Lab_target, *d_CIECAM97s_source, *d_CIECAM97s_target;
	float* d_RGB_result, *d_Lab_result, *d_CIECAM97s_result;
	float* d_RGB_result_in_BGR, *d_Lab_result_in_BGR, *d_CIECAM97s_result_in_BGR;
	cudaMalloc(&d_BGR_source, 3 * sizeof(float) * source_rows * source_cols);
	cudaMalloc(&d_RGB_source, 3 * sizeof(float) * source_rows * source_cols);
	cudaMalloc(&d_BGR_target, 3 * sizeof(float) * target_rows * target_cols);
	cudaMalloc(&d_RGB_target, 3 * sizeof(float) * target_rows * target_cols);
	cudaMalloc(&d_Lab_source, 3 * sizeof(float) * source_rows * source_cols);
	cudaMalloc(&d_Lab_target, 3 * sizeof(float) * target_rows * target_cols);
	cudaMalloc(&d_CIECAM97s_source, 3 * sizeof(float) * source_rows * source_cols);
	cudaMalloc(&d_CIECAM97s_target, 3 * sizeof(float) * target_rows * target_cols);
	cudaMalloc(&d_RGB_result, 3 * sizeof(float) * source_rows * source_cols);
	cudaMalloc(&d_Lab_result, 3 * sizeof(float) * source_rows * source_cols);
	cudaMalloc(&d_CIECAM97s_result, 3 * sizeof(float) * source_rows * source_cols);
	cudaMalloc(&d_RGB_result_in_BGR, 3 * sizeof(float) * source_rows * source_cols);
	cudaMalloc(&d_Lab_result_in_BGR, 3 * sizeof(float) * source_rows * source_cols);
	cudaMalloc(&d_CIECAM97s_result_in_BGR, 3 * sizeof(float) * source_rows * source_cols);
//Allocate Space for result
	float* h_RGB_result, *h_Lab_result, *h_CIECAM97s_result;
	h_RGB_result = (float*)malloc(3 * sizeof(float) * source_rows * source_cols);
	h_Lab_result = (float*)malloc(3 * sizeof(float) * source_rows * source_cols);
	h_CIECAM97s_result = (float*)malloc(3 * sizeof(float) * source_rows * source_cols);
//Copy Memory To Device
	cudaMemcpy(d_BGR_source, (float*)img_BGR_source.data, 3 * sizeof(float) * source_rows * source_cols, cudaMemcpyHostToDevice);	
	cudaMemcpy(d_BGR_target, (float*)img_BGR_target.data, 3 * sizeof(float) * target_rows * target_cols, cudaMemcpyHostToDevice);
//Convert BGR to RGB
	int source_threads = 3 * source_rows * source_cols;
	int target_threads = 3 * target_rows * target_cols;
	int num_source_blocks = ceil(source_threads/((float)BLOCK_DIM));
	int num_target_blocks = ceil(target_threads/((float)BLOCK_DIM));

	convert_color_space_BGR_to_RGB<<<num_source_blocks, BLOCK_DIM >>>(d_BGR_source, d_RGB_source, source_threads);
	convert_color_space_BGR_to_RGB<<<num_target_blocks, BLOCK_DIM>>>(d_BGR_target, d_RGB_target, target_threads);
	convert_color_space_RGB_to_BGR<<<num_target_blocks, BLOCK_DIM>>>(d_RGB_target, d_BGR_target, target_threads);
	convert_color_space_RGB_to_BGR<<<num_source_blocks, BLOCK_DIM>>>(d_RGB_source, d_BGR_source, source_threads);

//Color Transfer for RGB
	Mat img_RGB_new_RGB(source_rows, source_cols, CV_32FC3);
	color_transfer(d_RGB_source, d_RGB_target, d_RGB_result, temp_array, std::string("in_RGB"), source_threads, target_threads);
	convert_color_space_RGB_to_BGR<<<num_source_blocks, BLOCK_DIM>>>(d_RGB_result, d_RGB_result_in_BGR, source_threads);
	cudaMemcpy(h_RGB_result, d_RGB_result_in_BGR, source_threads * sizeof(float), cudaMemcpyDeviceToHost);	
	std::memcpy(img_RGB_new_RGB.data, h_RGB_result, source_threads * sizeof(float));
	imwrite(argv[3], img_RGB_new_RGB);
//Color Transfer for Lab
	
	Mat img_RGB_new_Lab(source_rows, source_cols, CV_32FC3);
	color_transfer(d_RGB_source, d_RGB_target, d_Lab_result, temp_array, std::string("in_Lab"), source_threads, target_threads);
	convert_color_space_RGB_to_BGR<<<num_source_blocks, BLOCK_DIM>>>(d_Lab_result, d_Lab_result_in_BGR, source_threads);
	cudaMemcpy(h_Lab_result, d_Lab_result_in_BGR, source_threads * sizeof(float), cudaMemcpyDeviceToHost);
	std::memcpy(img_RGB_new_Lab.data, h_Lab_result, source_threads * sizeof(float));
	imwrite(argv[4], img_RGB_new_Lab);
//Color Transfer for CIECAM97s
	Mat img_RGB_new_CIECAM97s(source_rows, source_cols, CV_32FC3);
	color_transfer(d_RGB_source, d_RGB_target, d_CIECAM97s_result, temp_array, std::string("in_CIECAM97s"), source_threads, target_threads);	
	convert_color_space_RGB_to_BGR<<<num_source_blocks, BLOCK_DIM>>>(d_CIECAM97s_result, d_CIECAM97s_result_in_BGR, source_threads);
	cudaMemcpy(h_CIECAM97s_result, d_CIECAM97s_result_in_BGR, source_threads * sizeof(float), cudaMemcpyDeviceToHost);
	std::memcpy(img_RGB_new_CIECAM97s.data, h_CIECAM97s_result, source_threads * sizeof(float));
	imwrite(argv[5], img_RGB_new_CIECAM97s);

//Deallocate Memory
	free(h_RGB_result);
	free(h_Lab_result);
	free(h_CIECAM97s_result);
	cudaFree(&d_BGR_source);
	cudaFree(&d_BGR_target);
	cudaFree(&d_RGB_source);
	cudaFree(&d_RGB_target);
	cudaFree(&d_Lab_source);
	cudaFree(&d_Lab_target);
	cudaFree(&d_CIECAM97s_source);
	cudaFree(&d_CIECAM97s_target);
	cudaFree(&temp_array);
	cudaFree(&d_RGB_result);
	cudaFree(&d_Lab_result);
	cudaFree(&d_CIECAM97s_result);
	cudaFree(&d_RGB_result_in_BGR);
	cudaFree(&d_Lab_result_in_BGR);
	cudaFree(&d_CIECAM97s_result_in_BGR);
	return 0;

	
}	
