#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "globals.h"

extern float r4_uniform_01 ( int *seed ); 	// Normal distribution function
extern unsigned int random_uint(unsigned int limit); // Random integer in [0, limit)
extern int seed;

__global__ void init_grid (CELL_TYPE (*grid)[N]);
__global__ void init_grid_empty_cells (CELL_TYPE (*grid)[N], curandState (*devStates)[N]); 
__global__ void init_grid_infected (CELL_TYPE (*grid)[N], COUNT_TYPE init_infected, int *number_infected, curandState (*devStates)[N]);
__global__ void kernel_init_random (int seed, curandState (*devStates)[N], PROBABILITY_TYPE (*pa)[N], PROBABILITY_TYPE (*f3)[N], PROBABILITY_TYPE (*RC)[N]);

__global__  void calculate_infections (CELL_TYPE (*grid)[N], PROBABILITY_TYPE (*pb)[N], PROBABILITY_TYPE (*f3)[N], PROBABILITY_TYPE (*RC)[N], curandState (*devStates)[N]);
__global__  void calculate_infection_transitions (CELL_TYPE (*grid)[N], PROBABILITY_TYPE (*pb)[N], PROBABILITY_TYPE (*pa)[N], curandState (*devStates)[N]);
__global__  void calculate_rest_state_transitions (CELL_TYPE (*grid)[N], CELL_TYPE (*time)[N][N], int t, PROBABILITY_TYPE uu, PROBABILITY_TYPE kk, COUNT_TYPE *isC, COUNT_TYPE *isH, COUNT_TYPE *isD, COUNT_TYPE *isI, COUNT_TYPE *isIa, COUNT_TYPE *isR, curandState (*devStates)[N]);
__global__  void copy_matrix_cell_type (CELL_TYPE (*grid)[N], CELL_TYPE (*grid_copy)[N] );
__global__  void copy_matrix_probability_type (PROBABILITY_TYPE (*m)[N], PROBABILITY_TYPE (*m_copy)[N]);
__global__  void simulate_population_movements_step1 (CELL_TYPE (*grid)[N], PROBABILITY_TYPE (*f3)[N], PROBABILITY_TYPE (*RC)[N], PROBABILITY_TYPE (*pa)[N], CELL_TYPE (*grid_next)[N], PROBABILITY_TYPE (*f3_next)[N], PROBABILITY_TYPE (*RC_next)[N], PROBABILITY_TYPE (*pa_next)[N], POSITION_TYPE (*position)[N], PROBABILITY_TYPE move_proportion, CELL_TYPE max_distance, curandState (*devStates)[N], COUNT_TYPE *cells_not_moved_gpu);
__global__  void simulate_population_movements_step2 (CELL_TYPE (*grid)[N], PROBABILITY_TYPE (*f3)[N], PROBABILITY_TYPE (*RC)[N], PROBABILITY_TYPE (*pa)[N], CELL_TYPE (*grid_next)[N], PROBABILITY_TYPE (*f3_next)[N], PROBABILITY_TYPE (*RC_next)[N], PROBABILITY_TYPE (*pa_next)[N], POSITION_TYPE (*position)[N], PROBABILITY_TYPE move_proportion, CELL_TYPE max_distance, curandState (*devStates)[N]);
__global__ void calculate_self_isolation_state_transitions (CELL_TYPE (*grid)[N], AC_TYPE *AC, int t, curandState (*devStates)[N]);



__global__ void init_grid (CELL_TYPE (*grid)[N]) {
    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    int j = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (i<N && j<N) {
        grid[i][j] = S;
    }
}


__global__ void init_grid_empty_cells (CELL_TYPE (*grid)[N], curandState (*devStates)[N]) {
    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    int j = (blockDim.x * blockIdx.x) + threadIdx.x;
    PROBABILITY_TYPE p;
    curandState localState;

    if (i >=LOWER_LIMIT && i<UPPER_LIMIT && j >= LOWER_LIMIT && j<UPPER_LIMIT) {
        //p = curand_uniform(&devStates[i][j]);
        localState = devStates[i][j];
        p = curand_uniform(&localState);
        if ( p <= VACANCY_RATIO && p >0) {
            grid[i][j] = E;
        }
        //devStates[i][j] = localState;
    }
}


__global__ void init_grid_infected (CELL_TYPE (*grid)[N], COUNT_TYPE init_infected, int *number_infected, curandState (*devStates)[N]) {
    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    int j = (blockDim.x * blockIdx.x) + threadIdx.x;
    PROBABILITY_TYPE p;
    int old;
    curandState localState;

    if (i<N && j<N) {
        p = (1.0*init_infected)/(N*N);
        localState = devStates[i][j];
        if ( (curand_uniform(&localState) < p) && (*number_infected >= 0) && (grid[i][j] != I) ) {
            old = atomicSub (number_infected, 1);
            if (old > 0) 
                grid[i][j] = I;
        }            
        devStates[i][j] = localState;
    }
}



__global__ void kernel_init_random (int seed, curandState (*devStates)[N], PROBABILITY_TYPE (*pa)[N], PROBABILITY_TYPE (*f3)[N], PROBABILITY_TYPE (*RC)[N]) {
    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    int j = (blockDim.x * blockIdx.x) + threadIdx.x;
    unsigned int idx = j+i*N;
    curandState localState;


    if (i<N && j<N) {
        /**/
        curand_init ((seed << 10)*i + j, idx, 0, &(devStates[i][j]) );
        localState = devStates[i][j];
        pa[i][j] = curand_uniform (&localState);
        f3[i][j] = curand_uniform (&localState);
        RC[i][j] = curand_uniform (&localState);
        devStates[i][j] = localState;
        /**if (i== 0 && j==0) {
            printf ("kernel_init_random (0,0) pa: %f f3:%f RC: %f\n", pa[i][j], f3[i][j], RC[i][j]);
        }/**/
    }
}


__global__  void calculate_infections (CELL_TYPE (*grid)[N], PROBABILITY_TYPE (*pb)[N], PROBABILITY_TYPE (*f3)[N], PROBABILITY_TYPE (*RC)[N], curandState (*devStates)[N]) {
    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    int j = (blockDim.x * blockIdx.x) + threadIdx.x;
	float AA, BB;
	PROBABILITY_TYPE irc;	// inverse resistance/inmunity coefficient
	int ifx1, ifx2, ifx3, ifx4, ifx5, ifx6, ifx7, ifx8;
	int ifx11, ifx22, ifx33, ifx44, ifx55, ifx66, ifx77, ifx88;
	//CELL_TYPE (*grid)[N] = (CELL_TYPE (*)[N]) health_states_grid;
	//PROBABILITY_TYPE (*pa)[N] = (PROBABILITY_TYPE (*)[N]) pa_;
	//PROBABILITY_TYPE (*f3)[N] = (PROBABILITY_TYPE (*)[N]) f3_;
	//PROBABILITY_TYPE (*RC)[N] = (PROBABILITY_TYPE (*)[N]) RC_;

	if (i >= LOWER_LIMIT && i<UPPER_LIMIT) {
		if (j >= LOWER_LIMIT && j<UPPER_LIMIT) {
            pb[i][j] = 0;
			if (grid[i][j] == S) {
				// If a person is un-infected, check if neighbours are infected
				ifx1 = (grid[i-1][j-1] == I);
				ifx11 = (grid[i-1][j-1] == Ia);
				ifx2 = (grid[i-1][j] == I);
				ifx22 = (grid[i-1][j] == Ia);
				ifx3 = (grid[i-1][j+1] == I);
				ifx33 = (grid[i-1][j+1] == Ia);
				ifx4 = (grid[i][j-1] == I);
				ifx44 = (grid[i][j-1] == Ia);
				ifx5 = (grid[i][j+1] == I);
				ifx55 = (grid[i][j+1] == Ia);
				ifx6 = (grid[i+1][j-1] == I);
				ifx66 = (grid[i+1][j-1] == Ia);
				ifx7 = (grid[i+1][j] == I);
				ifx77 = (grid[i+1][j] == Ia);
				ifx8 = (grid[i+1][j+1] == I);
				ifx88 = (grid[i+1][j+1] == Ia);

				if (ifx1 || ifx11 || ifx2 || ifx22 || ifx3 || ifx33 || ifx4 || ifx44 || ifx5 || ifx55 || ifx6 || ifx66 || ifx7 || ifx77 || ifx8 || ifx88) {
					irc = 1-RC[i][j];
                    __syncthreads();
					AA = sqrt(f3[i-1][j]*ifx2*irc) + sqrt(f3[i][j+1]*ifx5*irc) + 
						 sqrt(f3[i+1][j]*ifx7*irc) + sqrt(f3[i][j-1]*ifx4*irc) + 
						 sqrt(0.5*f3[i-1][j]*ifx22*irc) + sqrt(0.5*f3[i][j+1]*ifx55*irc) + 
						 sqrt(0.5*f3[i+1][j]*ifx77*irc) + sqrt(0.5*f3[i][j-1]*ifx44*irc);
					BB = sqrt(f3[i-1][j-1]*ifx1*irc) + sqrt(f3[i-1][j+1]*ifx3*irc) + 
						 sqrt(f3[i+1][j-1]*ifx6*irc) + sqrt(f3[i+1][j+1]*ifx8*irc) + 
						 sqrt(0.5*f3[i-1][j-1]*ifx11*irc) + sqrt(0.5*f3[i-1][j+1]*ifx33*irc) + 
						 sqrt(0.5*f3[i+1][j-1]*ifx66*irc) + sqrt(0.5*f3[i+1][j+1]*ifx88*irc);
					// Equation 3: probabilty of being infected.
					//Pij = AA *aa /4 + BB *bb /4 ;
					pb[i][j] = AA*(0.5)/4.0 + BB*(1.0/(2*sqrt(2.0)))/4.0;
				} else {
					//pb[i][j] = 0;
				}
			}
		}
	}
}


__global__  void calculate_infection_transitions (CELL_TYPE (*grid)[N], PROBABILITY_TYPE (*pb)[N], PROBABILITY_TYPE (*pa)[N], curandState (*devStates)[N]) {
    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    int j = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (i >= LOWER_LIMIT && i<UPPER_LIMIT) {
		if (j >= LOWER_LIMIT && j<UPPER_LIMIT) {
			if (grid[i][j] == S) {
				// The states of one individual in the cells are updated by following:
				if ( pb[i][j] > curand_uniform(&devStates[i][j]) && pa[i][j] > curand_uniform(&devStates[i][j]) ) {
					grid[i][j] = Ia;
				} else if (pb[i][j] > curand_uniform(&devStates[i][j]) && pa[i][j] <= curand_uniform(&devStates[i][j]) ) {
					grid[i][j] = I;
				}
            }
        }
    }
}



__global__  void calculate_rest_state_transitions (CELL_TYPE (*grid)[N], CELL_TYPE (*time)[N][N], int t, PROBABILITY_TYPE uu, PROBABILITY_TYPE kk, COUNT_TYPE *isC, COUNT_TYPE *isH, COUNT_TYPE *isD, COUNT_TYPE *isI, COUNT_TYPE *isIa, COUNT_TYPE *isR, curandState (*devStates)[N]) {

    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    int j = (blockDim.x * blockIdx.x) + threadIdx.x;
	//CELL_TYPE (*grid)[N] = (CELL_TYPE (*)[N]) health_states_grid;
	//CELL_TYPE (*time)[N][N] = (CELL_TYPE (*)[N][N]) time_;
	// In Octave code 'time1' is now matrix time[0], 'time2' matrix is time[1], 
	// 'time3' is time[2], and 'time4' is time[3]. 
	
	if (i >= LOWER_LIMIT && i<UPPER_LIMIT) {
		if (j >= LOWER_LIMIT && j<UPPER_LIMIT) {
			if (grid[i][j] == Ia) {
				//isIa[t] = isIa[t] + 1; 		
				atomicAdd (&isIa[t], 1); 		// Current infected (asymptomatic) number in simulation
				time[3][i][j] = time[3][i][j] + 1;
			}
			
			if (time[3][i][j] == T1+T2) {
				grid[i][j] = R;
				time[3][i][j] = 0;
			}
			
			if (grid[i][j] == I) {
				//isI[t] = isI[t] + 1; 		
				atomicAdd (&isI[t], 1); 		// Current infected (symptomatic) number in simulation
				time[0][i][j] = time[0][i][j] + 1;
			}
			
			if (time[0][i][j] == T1) {
				grid[i][j] = C;
				time[0][i][j] = 0;
				//isC[t] = isC[t] + 1; 		
				atomicAdd (&isC[t], 1); 		// Daily confirmed in simulation
			}
			
			if (grid[i][j] == C) {
				time[1][i][j] = time[1][i][j] + 1;
			}

			
			if (time[1][i][j] == T2) {
				if (curand_uniform(&devStates[i][j]) > uu) {
					grid[i][j] = R;
				} else {
					grid[i][j] = H;
					//isH[t] = isH[t] + 1;	
					atomicAdd (&isH[t], 1);	// Daily hospitalized in simulation
				}
				time[1][i][j] = 0;
			}
				
			if (grid[i][j] == H) {
				time[2][i][j] = time[2][i][j] + 1;
			}
			
			if (time[2][i][j] == T3) {
				if (curand_uniform(&devStates[i][j]) > kk) {
					grid[i][j] = R;
				} else {
					grid[i][j] = D;
					//isD[t] = isD[t] + 1;	
					atomicAdd (&isD[t], 1);	// Daily dead in simulation
				}
				time[2][i][j] = 0;
			}
			
			if (grid[i][j] == R) {
				//isR[t] = isR[t] + 1; 		
				atomicAdd (&isR[t], 1); 		// Cumulative recovered number in simulation
			}
		}
	}
}


__global__  void copy_matrix_cell_type (CELL_TYPE (*grid)[N], CELL_TYPE (*grid_copy)[N] ) {
    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    int j = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (i<N && j<N) {
        grid_copy[i][j] = grid[i][j];
    }
}


__global__  void copy_matrix_probability_type (PROBABILITY_TYPE (*m)[N], PROBABILITY_TYPE (*m_copy)[N]) {
    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    int j = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (i<N && j<N) {
        m_copy[i][j] = m[i][j];
    }
}



__global__  void simulate_population_movements_step1 (CELL_TYPE (*grid)[N], PROBABILITY_TYPE (*f3)[N], PROBABILITY_TYPE (*RC)[N], PROBABILITY_TYPE (*pa)[N], CELL_TYPE (*grid_next)[N], PROBABILITY_TYPE (*f3_next)[N], PROBABILITY_TYPE (*RC_next)[N], PROBABILITY_TYPE (*pa_next)[N], POSITION_TYPE (*position)[N], PROBABILITY_TYPE move_proportion, CELL_TYPE max_distance, curandState (*devStates)[N], COUNT_TYPE *cells_not_moved_gpu) {

    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    int j = (blockDim.x * blockIdx.x) + threadIdx.x;
	int t1, t2, di,dj;
    POSITION_TYPE pos;
    //int attempts = 0; 
    //bool end = 0;

	if (i>=(LOWER_LIMIT+max_distance) && i< (UPPER_LIMIT-max_distance) && j>=(LOWER_LIMIT+max_distance) && j< (UPPER_LIMIT-max_distance)) {
        if (curand_uniform(&devStates[i][j]) > 1-move_proportion) { 
            if (atomicCAS (&position[i][j], 0, NA) == 0) {
                // position[i][j] is free at this point. No other cell is going to use it for swapping. 
                //do {
				    // Swap 
                    t1 = curand_uniform(&devStates[i][j])*max_distance;
                    t1 = 1 + t1;
                    t2 = curand_uniform(&devStates[i][j])*max_distance;
                    t2 = 1 + t2;
                    di = t1 - t2;
                    t1 = curand_uniform(&devStates[i][j])*max_distance;
                    t1 = 1 + t1;
                    t2 = curand_uniform(&devStates[i][j])*max_distance;
                    t2 = 1 + t2;
                    dj = t1 - t2;
                    // (i,j) coordinates are recorded in "position[i+di][j+dj]".
                    // "pos" contains the "i" coordinate in the (16) upper bits and the "j" coordinate in the (16) lowest bits. 
                    pos = (i<<(sizeof(POSITION_TYPE)*8/2)) | (j & 0x0000FFFF);
			        // Do swapping (step 1)
                    if (atomicCAS (&position[i+di][j+dj], 0, pos) == 0) {
                        // position[i+di][j+dj] is free at this point. No other cell is going to use it for swapping. 
					    grid_next[i][j] = grid[i+di][j+dj];										
					    pa_next[i][j] = pa[i+di][j+dj];
					    f3_next[i][j] = f3[i+di][j+dj];
					    RC_next[i][j] = RC[i+di][j+dj];
                        //end = 1;										
                    } else {
                        atomicAdd (cells_not_moved_gpu,1);
                    }
                //} while (!end && attempts < max_distance);
			} else {
                atomicAdd (cells_not_moved_gpu,1);
            }
		}
	}
}




__global__  void simulate_population_movements_step2 (CELL_TYPE (*grid)[N], PROBABILITY_TYPE (*f3)[N], PROBABILITY_TYPE (*RC)[N], PROBABILITY_TYPE (*pa)[N], CELL_TYPE (*grid_next)[N], PROBABILITY_TYPE (*f3_next)[N], PROBABILITY_TYPE (*RC_next)[N], PROBABILITY_TYPE (*pa_next)[N], POSITION_TYPE (*position)[N], PROBABILITY_TYPE move_proportion, CELL_TYPE max_distance, curandState (*devStates)[N]) {

    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    int j = (blockDim.x * blockIdx.x) + threadIdx.x;
    int di,dj;
    POSITION_TYPE pos;

	if (i<N && j<N ) {
        pos = position[i][j];
        if (pos == NA) {
            position[i][j] = 0;    
        } else if (pos != 0) {
            // Do swapping (step 2)
            di = (pos >> (sizeof(POSITION_TYPE)*8/2)) & 0x0000FFFF;
            dj = pos & 0x0000FFFF;
            grid_next[i][j] = grid[di][dj];
            pa_next[i][j] = pa[di][dj];
            f3_next[i][j] = f3[di][dj];
            RC_next[i][j] = RC[di][dj];
            position[i][j] = 0;    
        }
    }
}




__global__ void calculate_self_isolation_state_transitions (CELL_TYPE (*grid)[N], AC_TYPE *AC, int t, curandState (*devStates)[N]) {
    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    int j = (blockDim.x * blockIdx.x) + threadIdx.x;
	PROBABILITY_TYPE q, qq;

	if (t>= 39) {
		q = 0.7 - 0.1*(AC[t-20]-AC[t-21])/AC[t-21]/0.025;
	    if (i >= LOWER_LIMIT && i<UPPER_LIMIT) {
		    if (j >= LOWER_LIMIT && j<UPPER_LIMIT) {
				// self-isolation proportion q 
				// correspond to Eq.(5)
				if (grid[i][j] == Si) {
					grid[i][j] = S;
				}
				qq = curand_uniform(&devStates[i][j]);
				if (qq <= q && qq > 0 && grid[i][j] == S) {
					grid[i][j] = Si;
				}
			}
		}
	}
}



