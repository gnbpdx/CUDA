#include <cuda.h>
#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h> // for printf
#include <time.h> // for nanosleep
#include <pthread.h>
#define WIDTH 60
#define HEIGHT 40
#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE + 2)
#define THREAD_COUNT 20
const int offsets[8][2] = {{-1, 1},{0, 1},{1, 1},
    {-1, 0},       {1, 0},
    {-1,-1},{0,-1},{1,-1}};

__device__ const int d_offsets[8][2] = {{-1, 1},{0, 1},{1, 1},
    {-1, 0},       {1, 0},
    {-1,-1},{0,-1},{1,-1}};
// Parameters passed in to pthread function:
struct args
{
	int width;
	int height;
	int* current;
	int* next;
	int index;
};

void fill_board(int *board, int width, int height) {
    int i;
    for (i=0; i<width*height; i++)
        board[i] = rand() % 2;
}

void print_board(int *board) {
    int x, y;
    for (y=0; y<HEIGHT; y++) {
        for (x=0; x<WIDTH; x++) {
            char c = board[y * WIDTH + x] ? '#':' ';
            printf("%c", c);
        }
        printf("\n");
    }
    printf("-----\n");
}
// Tests to see if two functions return the same output for validation
void test(int* table1, int* table2, int width, int height)
{
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		if (table1[i*width +j] != table2[i*width+j])
		{
			printf("Mismatch in step function at (%d,%d) Device: %d Host %d\n",  j, i, table1[i*width+j], table2[i*width+j]);
		}

	}
	printf("\n");
}
// Tiled version of CUDA step function. If block size is n x n, this creates an
// (n+2) x (n+2) array in shared memory that has stores all the values needed from global memory 
__global__ void d_step_tiled(int* current, int* next, int width, int height)
{
	// Put Variables into registers
	int threadx = threadIdx.x;
	int thready = threadIdx.y; 
	int blockx = blockIdx.x;
	int blocky = blockIdx.y;
	// Access row and column of thread
	int row = BLOCK_SIZE * blocky + thready;
	int col = BLOCK_SIZE * blockx + threadx;
	const int d_offsets[8][2] = {{-1, 1},{0, 1},{1, 1},
    {-1, 0},       {1, 0},
    {-1,-1},{0,-1},{1,-1}};
	// Set up shared memory
	__device__ __shared__ int tile[TILE_SIZE][TILE_SIZE];
	int nx, ny;
	// Fill shared memory with values

	ny = (row -1 + height) % height;	
	nx = (col - 1 + width) % width;
	tile[thready][threadx] = current[ny*width + nx];
	ny = (row + 1) % height;

	if (thready == BLOCK_SIZE -1 || thready == BLOCK_SIZE - 2)
		tile[thready + 2][threadx] = current[ny*width + nx];
	ny = (row -1 + height) % height;
	nx = (col + 1) % width;
	if (threadx == BLOCK_SIZE - 1 || threadx == BLOCK_SIZE - 2)

		tile[thready][threadx + 2] = current[ny*width + nx];
	ny = (row + 1) % height;
	if ((threadx == BLOCK_SIZE - 1 || threadx == BLOCK_SIZE - 2) && (thready == BLOCK_SIZE - 1 || thready == BLOCK_SIZE - 2))
		tile[thready + 2][threadx + 2] = current[ny * width + nx];
	__syncthreads();
	// Calculate neighbors
	if (row < height && col < width) {
	int num_neighbors = 0;
	for (int i = 0; i < 8; ++i) {
			int ny = (thready + 1) + d_offsets[i][0];
			int nx = (threadx + 1) + d_offsets[i][1];
			if (tile[ny][nx])
				++num_neighbors;
		}
	// Apply game of Life rules
	bool condition = (tile[thready + 1][threadx + 1] && num_neighbors == 2) || (num_neighbors == 3);
	next[row*width+col] = (int) condition;		
	}
}
// CUDA version of step function
__global__ void d_step(int* current, int* next, int width, int height)
{	
	// Put Variables into registers
	int threadx = threadIdx.x;
	int thready = threadIdx.y; 
	int blockx = blockIdx.x;
	int blocky = blockIdx.y;
	// Access row and column of thread
	int y = BLOCK_SIZE * blocky + thready;
	int x = BLOCK_SIZE * blockx + threadx;

	if (y < height && x < width) 
	{			
		int num_neighbors = 0;
		// Calculate number of neighbors
		for (int i = 0; i < 8; ++i)
		{
			int ny = (y + d_offsets[i][1] + height) % height;
			int nx = (x + d_offsets[i][0] + width) % width;
			if (current[ny*width + nx])
				++num_neighbors;
		}
		// Appy Game of Life Rules
		bool condition = (current[y*width+x] && (num_neighbors == 2)) || (num_neighbors == 3);
		next[y*width + x] = (int) condition;
	}
}
// P-thread version of step function. Pretty much a copy of the step function,
// except for calculating which pixels each thread is responsible for.
void *p_step(void* arguments)
{

  // coordinates of the cell we're currently evaluating
  // offset index, neighbor coordinates, alive neighbor count
  int i, nx, ny, num_neighbors, x,y;
	int index = ((struct args*)arguments)->index;
	int width = ((struct args*)arguments)->width;
	int height = ((struct args*)arguments)->height;
	int* current = ((struct args*)arguments)->current;
	int* next = ((struct args*)arguments)->next;
	int NUM_PIXELS = width * height;
	int start = (NUM_PIXELS * index) / THREAD_COUNT + 1;
	int end = (NUM_PIXELS * (index + 1)) / THREAD_COUNT;
	if (index == 0)
		start = 0;
	if (index == THREAD_COUNT - 1)
		end = NUM_PIXELS - 1;
	for (int p = start; p <= end; ++p) {
		y = p / width;
		x = p % width;
      // count this cell's alive neighbors
    	num_neighbors = 0;
      	for (i=0; i<8; i++) {
        // To make the board torroidal, we use modular arithmetic to
        // wrap neighbor coordinates around to the other side of the
        // board if they fall off.
        	nx = (x + offsets[i][0] + width) % width;
          ny = (y + offsets[i][1] + height) % height;
          if (current[ny * width + nx]) 
          	num_neighbors++;
          }
        // apply the Game of Life rules to this cell
        next[y * width + x] = 0;
        if ((current[y * width + x] && num_neighbors==2) || num_neighbors==3) 
          next[y * width + x] = 1;
	}
	return NULL;
}
void step(int *current, int *next, int width, int height) {
    // coordinates of the cell we're currently evaluating
    int x, y;
    // offset index, neighbor coordinates, alive neighbor count
    int i, nx, ny, num_neighbors;

    // write the next board state
    for (y=0; y<height; y++) {
        for (x=0; x<width; x++) {
 
         

            // count this cell's alive neighbors
            num_neighbors = 0;
            for (i=0; i<8; i++) {
                // To make the board torroidal, we use modular arithmetic to
                // wrap neighbor coordinates around to the other side of the
                // board if they fall off.
                nx = (x + offsets[i][0] + width) % width;
                ny = (y + offsets[i][1] + height) % height;
                if (current[ny * width + nx]) {
                    num_neighbors++;
                }

            }

            // apply the Game of Life rules to this cell
            next[y * width + x] = 0;
            if ((current[y * width + x] && num_neighbors==2) ||
                    num_neighbors==3) {
                next[y * width + x] = 1;
            }
        }
    }
}

int main(int argc, const char *argv[]) {
    // parse the width and height command line arguments, if provided
    int width, height, iters, out;
    if (argc < 3) {
        printf("usage: life iterations 1=print"); 
        exit(1);
    }
    iters = atoi(argv[1]);
    out = atoi(argv[2]);
    if (argc == 5) {
        width = atoi(argv[3]);
        height = atoi(argv[4]);
        printf("Running %d iterations at %d by %d pixels.\n", iters, width, height);
    } else {
        width = WIDTH;
        height = HEIGHT;
    }

    struct timespec delay = {0, 125000000}; // 0.125 seconds
    struct timespec remaining;
    // The two boards 
    int *current, *next, *d_current, *d_next, *d_tiled_current, *d_tiled_next, many=0;
		cudaEvent_t start, stop;
		float elapsed_time_ms;
    size_t board_size = sizeof(int) * width * height;
    current = (int *) malloc(board_size); // same as: int current[width * height];
    next = (int *) malloc(board_size);    // same as: int next[width *height];
		int* temp = (int*)malloc(board_size);
		int* p_current = (int*)malloc(board_size);
		int* p_next = (int*)malloc(board_size);
		cudaMalloc((void**)&d_current, board_size);
		cudaMalloc((void**)&d_next, board_size);
		cudaMalloc((void**)&d_tiled_current, board_size);	
		cudaMalloc((void**)&d_tiled_next, board_size);
		dim3 DimGrid(ceil(width/(float)BLOCK_SIZE),ceil(height/(float)BLOCK_SIZE),1);
		dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    // Initialize the global "current".
    fill_board(current, width, height);
		cudaMemcpy(d_current, current, board_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_tiled_current, current, board_size, cudaMemcpyHostToDevice);
		memcpy(p_current, current, board_size);
		cudaEventCreate(&start);
		cudaEventCreate(&stop);		
		pthread_t* threads = (pthread_t*) malloc(THREAD_COUNT*sizeof(pthread_t));
    while (many<iters) {
        many++;
        if (out==1)
            print_board(current);
				printf("Iteration %d:\n", many);	
        //evaluate the `current` board, writing the next generation into `next`.
				cudaEventRecord(start, 0);
        step(current, next, width, height);
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsed_time_ms, start, stop);
				printf("CPU Time: %f ms.\n", elapsed_time_ms);	
					
        // Copy the next state, that step() just wrote into, to current state
        memcpy(current, next, board_size);

        // copy the `next` to CPU and into `current` to be ready to repeat the process
				//CUDA step function	
				cudaEventRecord(start, 0);
				d_step<<<DimGrid, DimBlock>>>(d_current, d_next, width, height);
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsed_time_ms, start, stop);
				printf("Device Time: %f ms.\n", elapsed_time_ms);	

				//Useful Debugging Code:
				//cudaMemcpy(temp, d_next, board_size, cudaMemcpyDeviceToHost);
				//test(temp, next, width, height);	

				//Tiled version of CUDA step-function
				cudaEventRecord(start, 0);
				d_step_tiled<<<DimGrid, DimBlock>>>(d_tiled_current, d_tiled_next, width, height);
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsed_time_ms, start, stop);
				printf("Device Time for Tiled: %f ms.\n", elapsed_time_ms);	

				//Useful Debugging code:
				//cudaMemcpy(temp, d_tiled_next, board_size, cudaMemcpyDeviceToHost);
				//test(temp, next, width, height);

				//Pthread Step function:

				cudaEventRecord(start, 0);
				struct args* arguments = (struct args*)malloc(THREAD_COUNT*sizeof(struct args));	
				for(int i = 0; i < THREAD_COUNT; ++i)
				{
					arguments[i].width = width;
					arguments[i].height = height;
					arguments[i].current = p_current;
					arguments[i].next = p_next;
					arguments[i].index = i;
					pthread_create(threads + i, NULL, p_step, (void*)(arguments + i));
				}
				for (int i = 0; i < THREAD_COUNT; ++i)
				{
					pthread_join(*(threads + i), NULL);
				}
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsed_time_ms, start, stop);
				printf("Pthreads Time: %f ms.\n", elapsed_time_ms);	
				//Useful Debugging code:
				//test(p_next, next, width, height);	
				
				// Swap pointers (No need to copy data)
				
				int* temp2 = d_current;
				d_current = d_next;
				d_next = temp2;  	

				temp2 = d_tiled_current;
				d_tiled_current = d_tiled_next;
				d_tiled_next = temp2;
				
				temp2 = p_current;
				p_current = p_next;
				p_next = temp2;
        // We sleep only because textual output is slow and the console needs
        // time to catch up. We don't sleep in the graphical X11 version.
        if (out==1)
            nanosleep(&delay, &remaining);
				puts("");
    }
	free(current);
	free(next);
	free(p_current);
	free(p_next);
	cudaFree(d_current);
	cudaFree(d_next);
	cudaFree(d_tiled_current);
	cudaFree(d_tiled_next);
    return 0;
}

