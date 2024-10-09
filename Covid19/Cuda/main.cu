/*
Adapted NY.m Octave code to C from paper:
Dai, Jindong; Zhai, Chi; Ai, Jiali; Ma, Jiaying; Wang, Jingde; Sun, Wei. 2021. 
"Modeling the Spread of Epidemics Based on Cellular Automata" Processes 9, no. 1: 55. https://doi.org/10.3390/pr9010055

This file is part of covid19_NY_main/cuda.

covid19_NY_main/cuda is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

covid19_NY_main/cuda is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with covid19_NY_main/cuda.  If not, see <https://www.gnu.org/licenses/>.

Copyright 2021 Daniel Cagigas-Muñiz (dcagigas@us.es)

Also: 	Copyright (c) 2001 Fabrice Bellard for the video processing part
		available at https://github.com/FFmpeg/FFmpeg/blob/master/doc/examples/encode_video.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <openssl/rand.h> 			// For random numbers. OpenSSL library must be installed.
#include <time.h>

#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "globals.h"



cudaError_t allocate_main_simulation_variables ();
void init_zero_grid (CELL_TYPE (*grid)[N]);
void init_grid_cpu (CELL_TYPE (*grid)[N]);	    // Use VACANCY_RATIO to set empty (E) cells
void init_infected (CELL_TYPE (*grid)[N]);

int load_daily_confirmed_data (AC_TYPE *AC, int max_lenght);	
												// File AC.txt MUST be in the
												// same folder as this program.
void set_hospitalization_and_dead_fraction (int t, PROBABILITY_TYPE *uu, PROBABILITY_TYPE *kk);
void swapp_matrices_step3 ();

__global__ void init_grid (CELL_TYPE (*grid)[N]);
__global__ void init_grid_empty_cells (CELL_TYPE (*grid)[N], curandState (*devStates)[N]); 
__global__ void init_grid_infected (CELL_TYPE (*grid)[N], COUNT_TYPE init_infected, int *number_infected, curandState (*devStates)[N]);
 __global__ void kernel_init_random (int seed, curandState (*devStates)[N], PROBABILITY_TYPE (*pa)[N], PROBABILITY_TYPE (*f3)[N], PROBABILITY_TYPE (*RC)[N]);
cudaError_t GetMem(void ** devPtr, size_t size);

__global__  void calculate_infections (CELL_TYPE (*grid)[N], PROBABILITY_TYPE (*pb)[N], PROBABILITY_TYPE (*f3)[N], PROBABILITY_TYPE (*RC)[N], curandState (*devStates)[N]);
__global__  void calculate_infection_transitions (CELL_TYPE (*grid)[N], PROBABILITY_TYPE (*pb)[N], PROBABILITY_TYPE (*pa)[N], curandState (*devStates)[N]);
__global__  void calculate_rest_state_transitions (CELL_TYPE (*grid)[N], CELL_TYPE (*time)[N][N], int t, PROBABILITY_TYPE uu, PROBABILITY_TYPE kk, COUNT_TYPE *isC, COUNT_TYPE *isH, COUNT_TYPE *isD, COUNT_TYPE *isI, COUNT_TYPE *isIa, COUNT_TYPE *isR, curandState (*devStates)[N]);
__global__  void copy_matrix_cell_type (CELL_TYPE (*grid)[N], CELL_TYPE (*grid_copy)[N] );
__global__  void copy_matrix_probability_type (PROBABILITY_TYPE (*m)[N], PROBABILITY_TYPE (*m_copy)[N]);
__global__  void simulate_population_movements_step1 (CELL_TYPE (*grid)[N], PROBABILITY_TYPE (*f3)[N], PROBABILITY_TYPE (*RC)[N], PROBABILITY_TYPE (*pa)[N], CELL_TYPE (*grid_next)[N], PROBABILITY_TYPE (*f3_next)[N], PROBABILITY_TYPE (*RC_next)[N], PROBABILITY_TYPE (*pa_next)[N], POSITION_TYPE (*position)[N], PROBABILITY_TYPE move_proportion, CELL_TYPE max_distance, curandState (*devStates)[N], COUNT_TYPE *cells_not_moved_gpu);
__global__  void simulate_population_movements_step2 (CELL_TYPE (*grid)[N], PROBABILITY_TYPE (*f3)[N], PROBABILITY_TYPE (*RC)[N], PROBABILITY_TYPE (*pa)[N], CELL_TYPE (*grid_next)[N], PROBABILITY_TYPE (*f3_next)[N], PROBABILITY_TYPE (*RC_next)[N], PROBABILITY_TYPE (*pa_next)[N], POSITION_TYPE (*position)[N], PROBABILITY_TYPE move_proportion, CELL_TYPE max_distance, curandState (*devStates)[N]);

__global__ void calculate_self_isolation_state_transitions (CELL_TYPE (*grid)[N], AC_TYPE *AC, int t, curandState (*devStates)[N]);


int write_results_to_file ();
void free_main_simulation_variables ();

// Auxiliary functions to get random numbers and probabilities:
float r4_uniform_01 ( int *seed ); 	// Normal distribution function
unsigned int random_uint(unsigned int limit); // Random integer in [0, limit)
long int count_cells (CELL_TYPE (*grid)[N], CELL_TYPE c);

#ifdef __VIDEO
	uint8_t* generate_rgb (int width, int height, CELL_TYPE (*grid)[N], uint8_t *rgb);
	void write_CA_screenshot_image (const char *filename, int width, int height, uint8_t *img);
#endif	

void print_matrix (CELL_TYPE (*m)[N], int start_i, int end_i, int start_j, int end_j, char name[]);


int seed = 1985; 	// Seed for random uniform number output.

// Declare main simulation variables
CELL_TYPE (*grid)[N];			// Main state matrix 
CELL_TYPE (*grid_next)[N];
AC_TYPE *AC;				    // Contain daily confirmed number in AC.txt
AC_TYPE *AC_cpu;				
CELL_TYPE (*grid_cpu)[N];			
PROBABILITY_TYPE (*pb)[N];		// Probability of being infected
CELL_TYPE (*timer)[N][N];		// Timers matrix (time1, time2, time3)
PROBABILITY_TYPE (*pa)[N];		// Probability of becoming asymptomatic
PROBABILITY_TYPE (*pa_next)[N];
PROBABILITY_TYPE (*f3)[N];		// Infectivity of each person
PROBABILITY_TYPE (*f3_next)[N];
PROBABILITY_TYPE (*RC)[N];		// Resistance / immunity to the infection
PROBABILITY_TYPE (*RC_next)[N];
POSITION_TYPE (*position)[N];   // Auxiliary matrix used to implement cell swapping in the population movement simulation phase 
COUNT_TYPE cells_not_moved_cpu = 0;
COUNT_TYPE *cells_not_moved_gpu;
int number_infected_cpu = 0;
int *number_infected_gpu;

// Vector to record statistics: daily confirmed, hospitalized and dead in simulation
COUNT_TYPE *isC;	// daily confirmed number in simulation
COUNT_TYPE *isH ;	// daily hospitalized number in simulation
COUNT_TYPE *isD;	// daily dead number in simulation
COUNT_TYPE *isI;	// current infected (symptomatic) number in simulation
COUNT_TYPE *isIa;	// current infected (asymptomatic) number in simulation
COUNT_TYPE *isR;	// culmulative recovered number in simulation

COUNT_TYPE *isC_cpu;	
COUNT_TYPE *isH_cpu ;	
COUNT_TYPE *isD_cpu;	
COUNT_TYPE *isI_cpu;	
COUNT_TYPE *isIa_cpu;	
COUNT_TYPE *isR_cpu;	

curandState (*devStates)[N];

#ifdef __VIDEO
	uint8_t *rgb = NULL;
#endif



int main (void) {
    int iter = 0;
    int t;
    PROBABILITY_TYPE uu;					// Hospitalization fraction
    PROBABILITY_TYPE kk;					// Dead fraction
    PROBABILITY_TYPE move_proportion;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE,1);
    int linGrid = (int)ceil(N/(float)BLOCK_SIZE);
    dim3 gridSize(linGrid,linGrid,1);
	//cudaEvent_t time_stamp1, time_stamp2;
	//float total_time;

    #ifdef __VIDEO
        char image_name[80];
        rgb= (uint8_t *)malloc (3 * sizeof(uint8_t) *N*N);	
    #endif

	//cudaEventCreate(&time_stamp1);
	//cudaEventCreate(&time_stamp2);

//    printf ("Allocating memory ...\n");
    allocate_main_simulation_variables ();

//    printf ("Init variables and grids  ...\n");
    kernel_init_random <<<gridSize, blockSize>>> (time(NULL), devStates, pa, f3, RC);
    cudaDeviceSynchronize();

/**    
    init_zero_grid (grid_cpu);
	init_grid_cpu (grid_cpu);
	init_infected (grid_cpu);
    cudaMemcpy(grid, grid_cpu, sizeof(CELL_TYPE)*N*N, cudaMemcpyHostToDevice);
/**/



/**/   
    init_grid <<<gridSize, blockSize>>> (grid);
    cudaDeviceSynchronize();
    init_grid_empty_cells <<<gridSize, blockSize>>> (grid, devStates);
    cudaDeviceSynchronize();

    bool end = 0;
    number_infected_cpu = INIT_INFECTED;
    do {
        cudaMemcpy(number_infected_gpu, &number_infected_cpu, sizeof(int), cudaMemcpyHostToDevice);
        init_grid_infected <<<gridSize, blockSize>>> (grid, INIT_INFECTED, number_infected_gpu, devStates);
        cudaDeviceSynchronize();
        cudaMemcpy(&number_infected_cpu, number_infected_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        if (number_infected_cpu <= 0)
            end = 1;
    } while (!end);

    cudaMemcpy(grid_cpu, grid, sizeof(CELL_TYPE)*N*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(&number_infected_cpu, number_infected_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    long int res = count_cells (grid_cpu, I);
//    printf ("Infected cells at init: %ld - number_infected_cpu %d\n", res, number_infected_cpu);
    res = count_cells (grid_cpu, E);
//    printf ("Empty cells at init: %ld\n", res);
/**/


	if (load_daily_confirmed_data(AC_cpu, AC_DATA_LENGHT)==-1) {
		// The "AC.txt" file is not in the same folder of the program.
		printf ("Error: The file %s doesn't exist\n", DAILY_CONFIRMED_FILE);
		return -1;
	}

    cudaMemcpy(AC, AC_cpu, sizeof(AC_TYPE)*AC_DATA_LENGHT, cudaMemcpyHostToDevice);

 
            //cudaMemcpy(grid_cpu, grid, sizeof(CELL_TYPE)*N*N, cudaMemcpyDeviceToHost);
			//rgb = generate_rgb (N, N, grid_cpu, rgb);
			//sprintf (image_name, "%s%d.ppm", "ca_step_", 0);
			//write_CA_screenshot_image (image_name, N, N, rgb);

    // printf ("Init main loop  ...\n");

	//cudaEventRecord( time_stamp1, 0 ); 
	//cudaEventSynchronize( time_stamp1 ); 

	// Main CA time (days) simulation 
	//for (t=1; t<=TIME_STEPS; t++) {
	for (t=1; t<=TIME_STEPS; t++) {
		//printf ("%d\n", t);
		//printf ("%d - %d\n", t, iter);

		set_hospitalization_and_dead_fraction (t, &uu, &kk);
        /**/
		calculate_infections  <<<gridSize, blockSize>>> (grid, pb, f3, RC, devStates);
        cudaDeviceSynchronize();
		calculate_infection_transitions  <<<gridSize, blockSize>>> (grid, pb, pa, devStates);
        cudaDeviceSynchronize();
        calculate_rest_state_transitions <<<gridSize, blockSize>>> (grid, timer, t, uu, kk, isC, isH, isD, isI, isIa, isR, devStates);
        cudaDeviceSynchronize();
        /**/
        copy_matrix_cell_type <<<gridSize, blockSize>>> (grid, grid_next);
        cudaDeviceSynchronize();
        copy_matrix_probability_type <<<gridSize, blockSize>>> (f3, f3_next);
        cudaDeviceSynchronize();
        copy_matrix_probability_type <<<gridSize, blockSize>>> (pa, pa_next);
        cudaDeviceSynchronize();
        copy_matrix_probability_type <<<gridSize, blockSize>>> (RC, RC_next);
        cudaDeviceSynchronize();

        iter = 0;
        move_proportion = MOVE_PROPORTION;
        do {
            cells_not_moved_cpu = 0;
            cudaMemcpy(cells_not_moved_gpu, &cells_not_moved_cpu, sizeof(COUNT_TYPE), cudaMemcpyHostToDevice);
            simulate_population_movements_step1 <<<gridSize, blockSize>>> (grid, f3, RC, pa, grid_next, f3_next, RC_next, pa_next, position, move_proportion, L, devStates, cells_not_moved_gpu);
            cudaDeviceSynchronize();
            simulate_population_movements_step2 <<<gridSize, blockSize>>> (grid, f3, RC, pa, grid_next, f3_next, RC_next, pa_next, position, move_proportion, L, devStates);
            cudaDeviceSynchronize();
            cudaMemcpy(&cells_not_moved_cpu, cells_not_moved_gpu, sizeof(COUNT_TYPE), cudaMemcpyDeviceToHost);
            move_proportion = cells_not_moved_cpu/((N-L-LOWER_LIMIT)*(N-L-LOWER_LIMIT));
            iter++;
        } while (cells_not_moved_cpu > 0);

        swapp_matrices_step3 ();

        calculate_self_isolation_state_transitions <<<gridSize, blockSize>>> (grid, AC, t, devStates);
        cudaDeviceSynchronize();
		#ifdef __VIDEO
		if (t==21 || t==40 || t==60 || t==80 || t==140 || t==199) {
		//if (t==199) {
            cudaMemcpy(grid_cpu, grid, sizeof(CELL_TYPE)*N*N, cudaMemcpyDeviceToHost);
			rgb = generate_rgb (N, N, grid_cpu, rgb);
			sprintf (image_name, "%s%d.ppm", "ca_step_", t);
			write_CA_screenshot_image (image_name, N, N, rgb);
            //printf ("%d - %ld\n", t, count_cells (grid_cpu, I));
		}
        #endif
    }

	//cudaEventRecord( time_stamp2, 0 ); 
	//cudaEventSynchronize( time_stamp2 );
	//cudaEventElapsedTime( &total_time, time_stamp1, time_stamp2); 
	//printf("Total computation time in GPU: %f ms\n", total_time);

    // Get the results and write them to file:
    cudaMemcpy(grid_cpu, grid, sizeof(CELL_TYPE)*N*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(isC_cpu, isC, sizeof(COUNT_TYPE)*(TIME_STEPS+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(isH_cpu, isH, sizeof(COUNT_TYPE)*(TIME_STEPS+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(isD_cpu, isD, sizeof(COUNT_TYPE)*(TIME_STEPS+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(isI_cpu, isI, sizeof(COUNT_TYPE)*(TIME_STEPS+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(isIa_cpu, isIa, sizeof(COUNT_TYPE)*(TIME_STEPS+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(isR_cpu, isR, sizeof(COUNT_TYPE)*(TIME_STEPS+1), cudaMemcpyDeviceToHost);
    
    write_results_to_file ();
    free_main_simulation_variables ();

    return OK;
}


cudaError_t allocate_main_simulation_variables () {
    cudaError_t cudaStatus;

	cudaStatus = GetMem((void**) &number_infected_gpu, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "number_infected_gpu cudaMalloc failed!");
        return cudaStatus;
    }  
    // devStates (for Random generation)
	cudaStatus = GetMem ((void**) &devStates, N*N*sizeof( curandState ) );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "devStates cudaMalloc failed!");
        return cudaStatus;
    }
    // Grid: main CA state
    grid_cpu = (CELL_TYPE (*)[N]) malloc(sizeof(CELL_TYPE)*N*N);
	cudaStatus = GetMem((void**) &grid, sizeof(CELL_TYPE)*N*N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "grid cudaMalloc failed!");
        return cudaStatus;
    }  
	cudaStatus = GetMem((void**) &grid_next, sizeof(CELL_TYPE)*N*N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "grid_next cudaMalloc failed!");
        return cudaStatus;
    }  
    // AC: actual data
    AC_cpu = (AC_TYPE *)malloc(sizeof(AC_TYPE)*AC_DATA_LENGHT);
	cudaStatus = GetMem((void**) &AC, sizeof(AC_TYPE)*AC_DATA_LENGHT);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "AC cudaMalloc failed!");
        return cudaStatus;
    }  
    // Timers:
	cudaStatus = GetMem((void**) &timer, sizeof(CELL_TYPE)*4*N*N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "timer cudaMalloc failed!");
        return cudaStatus;
    }  
    // pb:
	cudaStatus = GetMem((void**) &pb, sizeof(PROBABILITY_TYPE)*N*N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "pb cudaMalloc failed!");
        return cudaStatus;
    }  
    // pa:
	cudaStatus = GetMem((void**) &pa, sizeof(PROBABILITY_TYPE)*N*N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "pa cudaMalloc failed!");
        return cudaStatus;
    }  
	cudaStatus = GetMem((void**) &pa_next, sizeof(PROBABILITY_TYPE)*N*N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "pa_next cudaMalloc failed!");
        return cudaStatus;
    }  
    // f3:
	cudaStatus = GetMem((void**) &f3, sizeof(PROBABILITY_TYPE)*N*N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "f3 cudaMalloc failed!");
        return cudaStatus;
    }  
	cudaStatus = GetMem((void**) &f3_next, sizeof(PROBABILITY_TYPE)*N*N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "f3_next cudaMalloc failed!");
        return cudaStatus;
    }  
    // RC:
	cudaStatus = GetMem((void**) &RC, sizeof(PROBABILITY_TYPE)*N*N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "RC cudaMalloc failed!");
        return cudaStatus;
    }  
	cudaStatus = GetMem((void**) &RC_next, sizeof(PROBABILITY_TYPE)*N*N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "RC_next cudaMalloc failed!");
        return cudaStatus;
    }  
    // position:
	cudaStatus = GetMem((void**) &position, sizeof(POSITION_TYPE)*N*N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "position cudaMalloc failed!");
        return cudaStatus;
    }  
	cudaStatus = GetMem((void**) &cells_not_moved_gpu, sizeof(COUNT_TYPE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cells_not_moved_gpu cudaMalloc failed!");
        return cudaStatus;
    }  
    // isC
    isC_cpu = (COUNT_TYPE *)malloc(sizeof(COUNT_TYPE)*(TIME_STEPS+1));
	cudaStatus = GetMem((void**) &isC, sizeof(COUNT_TYPE)*(TIME_STEPS+1));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "isC cudaMalloc failed!");
        return cudaStatus;
    }  
    // isH
    isH_cpu = (COUNT_TYPE *)malloc(sizeof(COUNT_TYPE)*(TIME_STEPS+1));
	cudaStatus = GetMem((void**) &isH, sizeof(COUNT_TYPE)*(TIME_STEPS+1));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "isH cudaMalloc failed!");
        return cudaStatus;
    }  
    // isD
    isD_cpu = (COUNT_TYPE *)malloc(sizeof(COUNT_TYPE)*(TIME_STEPS+1));
	cudaStatus = GetMem((void**) &isD, sizeof(COUNT_TYPE)*(TIME_STEPS+1));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "isD cudaMalloc failed!");
        return cudaStatus;
    }  
    // isI
    isI_cpu = (COUNT_TYPE *)malloc(sizeof(COUNT_TYPE)*(TIME_STEPS+1));
	cudaStatus = GetMem((void**) &isI, sizeof(COUNT_TYPE)*(TIME_STEPS+1));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "isI cudaMalloc failed!");
        return cudaStatus;
    }  
    // isIa
    isIa_cpu = (COUNT_TYPE *)malloc(sizeof(COUNT_TYPE)*(TIME_STEPS+1));
	cudaStatus = GetMem((void**) &isIa, sizeof(COUNT_TYPE)*(TIME_STEPS+1));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "isIa cudaMalloc failed!");
        return cudaStatus;
    }  
    // isR
    isR_cpu = (COUNT_TYPE *)malloc(sizeof(COUNT_TYPE)*(TIME_STEPS+1));
	cudaStatus = GetMem((void**) &isR, sizeof(COUNT_TYPE)*(TIME_STEPS+1));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "isR cudaMalloc failed!");
        return cudaStatus;
    }  

    return cudaStatus;
}


void init_zero_grid (CELL_TYPE (*grid)[N]) {	
	// Init to zero (S)
	int i, j;
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			grid[i][j] = 0;
		}
	}
}



void init_grid_cpu (CELL_TYPE (*grid)[N]) {	
	// Use VACANCY_RATIO to set empty (E) cells
	//CELL_TYPE (*grid)[N] = (CELL_TYPE (*)[N]) health_states_grid;
	PROBABILITY_TYPE p;
	int i, j;
	
	for (i=LOWER_LIMIT; i<UPPER_LIMIT; i++) {
		for (j=LOWER_LIMIT; j<UPPER_LIMIT; j++) {
			p = r4_uniform_01(&seed);
			if ( p <= VACANCY_RATIO && p >0) {
				// Initial empty cells
				grid[i][j] = E;
			} 
		}
	}
}


void init_infected (CELL_TYPE (*grid)[N]) {	
	// Use VACANCY_RATIO to set empty (E) cells
	//CELL_TYPE (*grid)[N] = (CELL_TYPE (*)[N]) health_states_grid;
	int i, j, cont;
	for (cont=0; cont<INIT_INFECTED; cont++) {
		i = random_uint (N);
		j = random_uint (N);
		grid[i][j] = I;
	}
}



int load_daily_confirmed_data (AC_TYPE *AC_cpu, int max_lenght) {	
	// File AC.txt MUST be in the
	// same folder as this program.
	FILE *f;
	int max_len_comment = 160;
	char c, str[max_len_comment];
	char comment = '#';
	int i, cont = 1;
	AC_TYPE value;
	
	f = fopen (DAILY_CONFIRMED_FILE, "rt");
	if (f==NULL) {
		// This file doesn't exist. Simulation must stop.
		return -1;
	}
	
	AC_cpu[0] = 0;
	fgets(str,max_len_comment,f);		
	while (!feof(f) && cont<max_lenght) {
		sscanf (str, "%c", &c);
		if (c != comment && c != '\n' && strlen(str)>0) {
			// If it doesn't begin with '#' or '\n' is not a comment or a newline.
			sscanf (str, "%d", &value);
			AC_cpu[cont] = value;
			cont++;
		}
		fgets(str,max_len_comment,f);
	}
	// Just in case there could be in the future longer vectors (periods of time).
	for (i=cont; i<max_lenght; i++) {
		AC_cpu[i] = 0;
	}
	return 0;
}


void set_hospitalization_and_dead_fraction (int t, PROBABILITY_TYPE *uu, PROBABILITY_TYPE *kk) {
	if (t <= 70) {
		*uu = U_6_MARCH_TO_23_APRIL;
		*kk = K_6_MARCH_TO_23_APRIL;
	} else if (t <= 120) {
		*uu = U_24_APRIL_TO_12_JUNE;
		*kk = K_24_APRIL_TO_12_JUNE;
	} else if (t <= 170) {
		*uu = U_13_JUNE_TO_1_AUGUST;
		*kk = K_13_JUNE_TO_1_AUGUST;
	} else {
		*uu = U_AFTER_1_AUGUST;
		*kk = K_AFTER_1_AUGUST;		
	}
}
	

// Function that performs the grid, f3, RC, pa swapping. This is necessary for the population movement simulation.
void swapp_matrices_step3 () {
    CELL_TYPE (*temp1)[N];			
    PROBABILITY_TYPE (*temp2)[N];		

    temp1 = grid;
    grid = grid_next; 
    grid_next = temp1;

    temp2 = f3;
    f3 = f3_next; 
    f3_next = temp2;

    temp2 = RC;
    RC = RC_next; 
    RC_next = temp2;

    temp2 = pa;
    pa = pa_next; 
    pa_next = temp2;
}





int write_results_to_file () {
	FILE *f;
	char filename[60];
	struct tm *timenow;
	int i;
	
	time_t now = time(NULL);
	timenow = gmtime(&now);
	strftime(filename, sizeof(filename), "covid19_cuda_results_%Y-%m-%d_%H:%M:%S.txt", timenow);
	
	f = fopen(filename,"w");
	if (f == NULL) {
		printf ("Error, file %s can't be openned \n", filename);
		return -1;
	}
	
	fprintf (f,"# New York state Covid-19 epidemic data simulation. \n");
	fprintf (f,"# Simulation data involves %d days in year 2020. \n", TIME_STEPS-21);
	fprintf (f,"# From 6th March 2020 to 31st August\n");
	fprintf (f,"# Confirmed, Hospitalized, Dead, Infected, Asymptomatic, Recovered, Infected+Asymptomatic. \n\n");
	fprintf (f,"# \t C\t    H\t  D\t  I\t   Ia\t  R\t  I+Ia\n\n");
	
	for (i=21; i<=TIME_STEPS; i++) {
	//for (i=1; i<=TIME_STEPS; i++) {
		fprintf (f, "%6d %6d %6d %6d %6d %6d %6d \n", isC_cpu[i], isH_cpu[i], isD_cpu[i], isI_cpu[i], isIa_cpu[i], isR_cpu[i], isI_cpu[i]+isIa_cpu[i]);
	}

	return 0;
}


void free_main_simulation_variables () {
	if (AC_cpu != NULL)
		free (AC_cpu);				
	if (grid_cpu != NULL)
		free (grid_cpu);			
	if (isC_cpu != NULL)		
		free (isC_cpu);				// daily confirmed number in simulation
	if (isH_cpu != NULL)		
		free (isH_cpu);				// daily hospitalized number in simulation
	if (isD_cpu != NULL)		
		free (isD_cpu);				// daily dead number in simulation
	if (isI_cpu != NULL) 			
		free (isI_cpu); 			// current infected (symptomatic) number in simulation
	if (isIa_cpu != NULL)  	
		free (isIa_cpu);			// current infected (asymptomatic) number in simulation
	if (isR_cpu != NULL)  
		free (isR_cpu);				// cumulative recovered number in simulation

    if (number_infected_gpu!=NULL)
		cudaFree (number_infected_gpu);		
	if (AC != NULL)
		cudaFree (AC);				// Contain daily confirmed number in AC.txt
	if (grid != NULL)
		cudaFree (grid);			// Main state matrix 
	if (grid_next != NULL)
		cudaFree (grid_next);		
	if (timer != NULL)
		cudaFree (timer);			// Timers matrix (time1, time2, time3)
	if (pb != NULL)		
		cudaFree (pb);				// Probability of being infected
	if (pa != NULL)		
		cudaFree (pa);				// Probability of becoming asymptomatic
    if (pa_next != NULL)		
		cudaFree (pa_next);
	if (f3 != NULL)		
		cudaFree (f3);				// Infectivity of each person
	if (f3_next != NULL)		
		cudaFree (f3_next);
	if (RC != NULL)		
		cudaFree (RC);				// Resistance / immunity to the infection
	if (RC_next != NULL)		
		cudaFree (RC_next);
	if (position != NULL)		    // Auxiliary matrix used to implement cell swapping in the population movement simulation phase 
		cudaFree (position);
    if (cells_not_moved_gpu != NULL)
		cudaFree (cells_not_moved_gpu);

	if (isC != NULL)		
		cudaFree (isC);				// daily confirmed number in simulation
	if (isH != NULL)		
		cudaFree (isH);				// daily hospitalized number in simulation
	if (isD != NULL)		
		cudaFree (isD);				// daily dead number in simulation
	if (isI != NULL) 			
		cudaFree (isI); 			// current infected (symptomatic) number in simulation
	if (isIa != NULL)  	
		cudaFree (isIa);			// current infected (asymptomatic) number in simulation
	if (isR != NULL)  
		cudaFree (isR);				// cumulative recovered number in simulation
}


cudaError_t GetMem(void ** devPtr, size_t size) {
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc(devPtr, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto GetMemError;
    }
	cudaStatus = cudaMemset(*devPtr, 0, size);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
        goto GetMemError;
    }

    GetMemError:
    return cudaStatus;
}

/******************************************************************************/
// Used for random numbers:
float r4_uniform_01 ( int *seed )
	
	/******************************************************************************/
	/*
	https://people.sc.fsu.edu/~jburkardt/c_src/normal/normal.html
	Purpose:
	
	R4_UNIFORM_01 returns a unit pseudorandom R4.
	
	Discussion:
	
	This routine implements the recursion
	
	seed = 16807 * seed mod ( 2^31 - 1 )
	r4_uniform_01 = seed / ( 2^31 - 1 )
	
	The integer arithmetic never requires more than 32 bits,
	including a sign bit.
	
	If the initial seed is 12345, then the first three computations are
	
	Input     Output      R4_UNIFORM_01
	SEED      SEED
	
	12345   207482415  0.096616
	207482415  1790989824  0.833995
	1790989824  2035175616  0.947702
	
	Licensing:
	
	This code is distributed under the GNU LGPL license. 
	
	Modified:
	
	16 November 2004
	
	Author:
	
	John Burkardt
	
	Reference:
	
	Paul Bratley, Bennett Fox, Linus Schrage,
	A Guide to Simulation,
	Springer Verlag, pages 201-202, 1983.
	
	Pierre L'Ecuyer,
	Random Number Generation,
	in Handbook of Simulation
	edited by Jerry Banks,
	Wiley Interscience, page 95, 1998.
	
	Bennett Fox,
	Algorithm 647:
	Implementation and Relative Efficiency of Quasirandom
	Sequence Generators,
	ACM Transactions on Mathematical Software,
	Volume 12, Number 4, pages 362-376, 1986.
	
	Peter Lewis, Allen Goodman, James Miller,
	A Pseudo-Random Number Generator for the System/360,
	IBM Systems Journal,
	Volume 8, pages 136-143, 1969.
	
	Parameters:
	
	Input/output, int *SEED, the "seed" value.  Normally, this
	value should not be 0.  On output, SEED has been updated.
	
	Output, float R4_UNIFORM_01, a new pseudorandom variate, strictly between
	0 and 1.
	*/
{
	const int i4_huge = 2147483647;
	int k;
	float r;
	
	k = *seed / 127773;
	
	*seed = 16807 * ( *seed - k * 127773 ) - k * 2836;
	
	if ( *seed < 0 )
	{
		*seed = *seed + i4_huge;
	}
	/*
	Although SEED can be represented exactly as a 32 bit integer,
	it generally cannot be represented exactly as a 32 bit real number!
	*/
	r = ( float ) ( *seed ) * 4.656612875E-10;
	
	return r;
}


/* Random integer in [0, limit).
   The best way to generate random numbers in C is to use a third-party library.
   The standard C function rand() is usually not recommended.
   OpenSSL's RAND_bytes() seeds itself. 
*/
unsigned int random_uint(unsigned int limit) {
	union {
		unsigned int i;
		unsigned char c[sizeof(unsigned int)];
	} u;
	
	do {
		if (!RAND_bytes(u.c, sizeof(u.c))) {
			fprintf(stderr, "Can't get random bytes!\n");
			exit(1);
		}
	} while (u.i < (-limit % limit)); /* u.i < (2**size % limit) */
	return u.i % limit;
}


// Functions used to generate CA evolution video:
#ifdef __VIDEO
uint8_t* generate_rgb (int width, int height, CELL_TYPE (*grid)[N], uint8_t *rgb) {
	int x, y, cur;
	
	// Transform "grid" (CA states grid) into a similar rgb image format.
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			cur = 3 * (y * width + x);
			if (grid[y][x] == S) {
				// Un-infected: Dark Blue
				rgb[cur + 0] = 0;
				rgb[cur + 1] = 0;
				rgb[cur + 2] = 255;
			} else if (grid[y][x] == E) {
				// Empty cell: Blue
				rgb[cur + 0] = 0;
				rgb[cur + 1] = 128;
				rgb[cur + 2] = 255;
			} else if (grid[y][x] == Si) {
				// Self-isolated: light blue
				rgb[cur + 0] = 0;
				rgb[cur + 1] = 255;
				rgb[cur + 2] = 255;
			} else if (grid[y][x] == I) {
				// Infected: Green
				rgb[cur + 0] = 0;
				rgb[cur + 1] = 255;
				rgb[cur + 2] = 0;
			} else if (grid[y][x] == Ia) {
				// Asymptomatic: Light green
				rgb[cur + 0] = 128;
				rgb[cur + 1] = 255;
				rgb[cur + 2] = 0;
			} else if (grid[y][x] == C) {
				// Confirmed: Yellow
				rgb[cur + 0] = 255;
				rgb[cur + 1] = 255;
				rgb[cur + 2] = 0;
			} else if (grid[y][x] == R) {
				// Recovered: Orange
				rgb[cur + 0] = 255;
				rgb[cur + 1] = 128;
				rgb[cur + 2] = 0;
			} else if (grid[y][x] == H) {
				// Hospitalized: Red
				rgb[cur + 0] = 255;
				rgb[cur + 1] = 0;
				rgb[cur + 2] = 0;
			} else if (grid[y][x] == D) {
				// Dead: Brown
				rgb[cur + 0] = 102;
				rgb[cur + 1] = 0;
				rgb[cur + 2] = 0;
			} else {
				// Any other state: White
				rgb[cur + 0] = 255;
				rgb[cur + 1] = 255;
				rgb[cur + 2] = 255;
			}
		}
	}
	return rgb;
}


// Writes to disk a CA state screenshot.
void write_CA_screenshot_image (const char *filename, int width, int height, uint8_t *img)
{
	FILE *fp;
	//open file for output
	fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}
	
	//write the header file
	//image format
	fprintf(fp, "P6\n");
	
	//comments
	fprintf(fp, "# Created by %s\n","Covid19_NY_serial");
	
	//image size
	fprintf(fp, "%d %d\n", width, height);
	
	// rgb component depth
	fprintf(fp, "%d\n",255);
	
	// pixel data
	fwrite(img, 3 * width, height, fp);
	fclose(fp);
}

#endif


void print_matrix (CELL_TYPE (*m)[N], int start_i, int end_i, int start_j, int end_j, char name[]) {	
	// For Debug purposes
	int i,j;
	
	printf ("\n------------%s\n", name);
	for (i=start_i; i<=end_i; i++) {
		for (j=start_j; j<=end_j; j++) {
			printf ("%d ", (int)m[i][j]);
		}
		printf ("\n");
	}
	printf ("------------\n\n");
}


long int count_cells (CELL_TYPE (*m)[N], CELL_TYPE c) {
    int i,j;
    long int total = 0;

	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
            if ( m[i][j] == c ) {
                    total++;
            }
        }
    }
    return total;
}


