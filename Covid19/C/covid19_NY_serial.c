/*
Adapted NY.m Octave code to C from paper:
Dai, Jindong; Zhai, Chi; Ai, Jiali; Ma, Jiaying; Wang, Jingde; Sun, Wei. 2021. 
"Modeling the Spread of Epidemics Based on Cellular Automata" Processes 9, no. 1: 55. https://doi.org/10.3390/pr9010055

This file is part of covid19_NY_serial.

covid19_NY_serial is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

covid19_NY_serial is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with covid19_NY_serial.  If not, see <https://www.gnu.org/licenses/>.

Copyright 2024 Daniel Cagigas-Muñiz (dcagigas@us.es)

Also: 	Copyright (c) 2001 Fabrice Bellard for the video processing part
		available at https://github.com/FFmpeg/FFmpeg/blob/master/doc/examples/encode_video.c
		Copyright (c) 2004 John Burkardt for the normal distribution part
		available at https://people.sc.fsu.edu/~jburkardt/c_src/normal/normal.html
*/

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
//#include <openssl/rand.h> 			// For random numbers. OpenSSL library must be installed.

#include "covid19_NY_serial.h"
#ifdef __VIDEO
	//#include <opencv2/core.hpp>
	//#include <opencv2/highgui.hpp>
	//#include <opencv2/videoio/videoio.hpp>
	//#include <opencv2/core/core.hpp>
	//#include <opencv2/highgui/highgui.hpp>
	//#include <opencv2/videoio/videoio.hpp>
	//#include <opencv4/opencv2/core/core.hpp>
	//#include <opencv4/opencv2/highgui/highgui.hpp>
	//#include "opencv4/opencv2/core/core.hpp"
	//#include "opencv4/opencv2/highgui/highgui.hpp"
	//#include "opencv2/core/core.hpp"
	//#include "opencv2/highgui/highgui.hpp"
	//#include "opencv2/core/core_c.h"
	//#include "opencv2/highgui/highgui_c.h"
	#include <opencv2/core/core_c.h>
	#include <opencv2/highgui/highgui_c.h>
	#include <opencv2/videoio/videoio_c.h>
#endif

//using namespace cv;

void allocate_main_simulation_variables ();
void init_grid (CELL_TYPE *health_states_grid);	// Use VACANCY_RATIO to set empty (E) cells
void init_infected (CELL_TYPE *health_states_grid);
void init_zero_grid (CELL_TYPE *grid);
void init_zero_vectors (COUNT_TYPE *isC, COUNT_TYPE *isH, COUNT_TYPE *isD, COUNT_TYPE *isI, COUNT_TYPE *isIa, COUNT_TYPE *isR);
int load_daily_confirmed_data (AC_TYPE *AC, int max_lenght);	
												// File AC.txt MUST be in the
												// same folder as this program.
void init_pa_f3_RC (PROBABILITY_TYPE *pa, PROBABILITY_TYPE *f3, PROBABILITY_TYPE *RC, CELL_TYPE *health_states_grid);

void set_hospitalization_and_dead_fraction (int t, PROBABILITY_TYPE *uu, PROBABILITY_TYPE *kk);
void calculate_infections (CELL_TYPE *health_states_grid, PROBABILITY_TYPE *pb, PROBABILITY_TYPE *f3, PROBABILITY_TYPE *RC);
void calculate_infection_transitions (CELL_TYPE *health_states_grid, PROBABILITY_TYPE *pb, PROBABILITY_TYPE *pa);
void calculate_rest_state_transitions (CELL_TYPE *health_states_grid, CELL_TYPE *time_, int t, PROBABILITY_TYPE uu, PROBABILITY_TYPE kk, COUNT_TYPE *isC, COUNT_TYPE *isH, COUNT_TYPE *isD, COUNT_TYPE *isI, COUNT_TYPE *isIa, COUNT_TYPE *isR);
void simulate_population_movements (CELL_TYPE *health_states_grid, PROBABILITY_TYPE *pa, PROBABILITY_TYPE *f3, PROBABILITY_TYPE *RC, PROBABILITY_TYPE move_proportion, CELL_TYPE max_distance);
void calculate_self_isolation_state_transitions (CELL_TYPE *health_states_grid, AC_TYPE *AC, int t);

int write_results_to_file ();
void free_main_simulation_variables ();

// Auxiliary functions to get random numbers and probabilities:
float r4_uniform_01 (unsigned int *seed ); 	// Normal distribution function
unsigned int random_uint(unsigned int limit, unsigned int *seed ); // Random integer in [0, limit)
void print_vector (AC_TYPE *v, int start, int end);
void print_matrix (CELL_TYPE *m, int start, int end);

#ifdef __VIDEO
	uint8_t* generate_rgb (int width, int height, CELL_TYPE *grid_, uint8_t *rgb);
	void write_CA_screenshot_image (const char *filename, int width, int height, uint8_t *img);
#endif

unsigned int seed = 1; 	// Seed for random uniform number output.

// Declare main simulation variables
AC_TYPE *AC;				// Contain daily confirmed number in AC.txt
CELL_TYPE *grid;			// Main state matrix 
PROBABILITY_TYPE *pb;		// Probability of being infected
CELL_TYPE *timer;			// Timers matrix (time1, time2, time3)
PROBABILITY_TYPE *pa;		// Probability of becoming asymptomatic
PROBABILITY_TYPE *f3;		// Infectivity of each person
PROBABILITY_TYPE *RC;		// Resistance / immunity to the infection

PROBABILITY_TYPE uu;					// Hospitalization fraction
PROBABILITY_TYPE kk;					// Dead fraction

// Vector to record statistics: daily confirmed, hospitalized and dead in simulation
COUNT_TYPE *isC;	// daily confirmed number in simulation
COUNT_TYPE *isH ;	// daily hospitalized number in simulation
COUNT_TYPE *isD;	// daily dead number in simulation
COUNT_TYPE *isI;	// current infected (symptomatic) number in simulation
COUNT_TYPE *isIa;	// current infected (asymptomatic) number in simulation
COUNT_TYPE *isR;	// culmulative recovered number in simulation

char DAILY_CONFIRMED_FILE[] = "AC.txt";

#ifdef __VIDEO
	uint8_t *rgb = NULL;
#endif

int main () {

    //srand(time(NULL));
    seed = (int)time(NULL);

	int t;
	// Functions used to generate CA video:
	#ifdef __VIDEO
		IplImage *img = 0;
		CvVideoWriter *writer;
		char video_file_name[]="covid19_NY_video.avi";
		writer = cvCreateVideoWriter(video_file_name, CV_FOURCC('D', 'I', 'V', 'X') , 24 , cvSize(N, N), true);
		img = cvCreateImage(cvSize(N, N), IPL_DEPTH_8U, 3);
		char image_name[80];
	#endif	
	
	allocate_main_simulation_variables ();
	init_zero_grid (&timer[0]);
	init_zero_grid (&timer[1]);
	init_zero_grid (&timer[2]);
	init_zero_grid (&timer[3]);
	init_zero_grid (grid);
	init_grid (grid);
	init_infected (grid);
	init_zero_vectors (isC, isH, isD, isI, isIa, isR);
	if (load_daily_confirmed_data(AC, AC_DATA_LENGHT)==-1) {
		// The "AC.txt" file is not in the same folder of the program.
		printf ("Error: The file %s doesn't exist\n", DAILY_CONFIRMED_FILE);
		return -1;
	}
	init_pa_f3_RC (pa, f3, RC, grid);

	// Main CA time (days) simulation 
	for (t=1; t<=TIME_STEPS; t++) {
		//printf ("%d\n", t);
		set_hospitalization_and_dead_fraction (t, &uu, &kk);    // Update probability values of an infected for being hospitalized/dead based on "t".
		calculate_infections (grid, pb, f3, RC);                // Update probabilities of population (i.e. each grid cell) of being infected.
		calculate_infection_transitions (grid, pb, pa);         // Update infected population (grid cells) based on their probabilities of being infected.
		calculate_rest_state_transitions (grid, timer, t, uu, kk, isC, isH, isD, isI, isIa, isR);   // Update rest of population states (grid cells) based on timers: Recovered, Hospitalized, Dead, Confirmed and Asymptomatic.
		simulate_population_movements (grid, pa, f3, RC, MOVE_PROPORTION, L);   // Exchange a fraction of the population (grid cells).
		calculate_self_isolation_state_transitions (grid, AC, t);               // Set the population that decide to become "self isolated".
		#ifdef __VIDEO
		// Write some CA screenshots images:
		/**/if (t==21 || t==40 || t==60 || t==80 || t==140 || t==199) {
			rgb = generate_rgb (N, N, grid, rgb);
			sprintf (image_name, "%s%d.ppm", "ca_step_", t);
			write_CA_screenshot_image (image_name, N, N, rgb);
		}/**/
		// VIDEO WRITE:
		rgb = generate_rgb (N, N, grid, rgb);
		for( int y=0; y<img->height; y++ ) { 
			uchar* ptr = (uchar*) ( img->imageData + y * img->widthStep ); 
			for( int x=0; x<img->width; x++ ) { 
				int cur = 3 * (y * N + x);
				ptr[3*x+2] = rgb[cur+0]; //Set red (BGR format)
				ptr[3*x+1] = rgb[cur+1]; //Set green (BGR format)
				ptr[3*x+0] = rgb[cur+2]; //Set blue (BGR format)
			}
		}		
		cvWriteFrame( writer, img );
		#endif
	}
	
	#ifdef __VIDEO
		cvReleaseImage( &img );
		cvReleaseVideoWriter( &writer );
	#endif
		
	write_results_to_file ();
	free_main_simulation_variables ();
	
    return 0;
}


void allocate_main_simulation_variables () {
	size_t bytes;
	
	bytes = sizeof(CELL_TYPE)*(N)*(N); 
	grid = (CELL_TYPE *)malloc(bytes); 		// Allocate Grid state for initial setup. 
											// In the Octave code "grid" is named "X"
	bytes = sizeof(CELL_TYPE)*4*(N)*(N); 	// There are 4 timer matrices
	timer = (CELL_TYPE *)malloc(bytes); 	// The 4 matrix timers in the Octave code
										
	
	bytes = sizeof(AC_TYPE)*AC_DATA_LENGHT; // The "AC.txt" file contains 178 day data
	AC = (AC_TYPE *)malloc(bytes);
	
	bytes = sizeof(PROBABILITY_TYPE)*(N)*(N);
	pb = (PROBABILITY_TYPE *)malloc(bytes);
	pa = (PROBABILITY_TYPE *)malloc(bytes);
	f3 = (PROBABILITY_TYPE *)malloc(bytes);
	RC = (PROBABILITY_TYPE *)malloc(bytes);
	
	bytes = sizeof(COUNT_TYPE)*(TIME_STEPS+1); 	// Days are counted starting in day 1 not 0
	isC = (COUNT_TYPE *)malloc(bytes);
	isH = (COUNT_TYPE *)malloc(bytes);
	isD = (COUNT_TYPE *)malloc(bytes);
	isI = (COUNT_TYPE *)malloc(bytes);
	isIa = (COUNT_TYPE *)malloc(bytes);
	isR = (COUNT_TYPE *)malloc(bytes);
	#ifdef __VIDEO
		rgb= (uint8_t *)malloc(3 * sizeof(uint8_t) *N*N);
	#endif
}


void init_zero_grid (CELL_TYPE *grid_) {	
	// Init to zero
	CELL_TYPE (*grid)[N] = (CELL_TYPE (*)[N]) grid_;
	int i, j;
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			grid[i][j] = 0;
		}
	}
}


void init_grid (CELL_TYPE *health_states_grid) {	
	// Use VACANCY_RATIO to set empty (E) cells
	CELL_TYPE (*grid)[N] = (CELL_TYPE (*)[N]) health_states_grid;
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


void init_infected (CELL_TYPE *health_states_grid) {	
	// Use VACANCY_RATIO to set empty (E) cells
	CELL_TYPE (*grid)[N] = (CELL_TYPE (*)[N]) health_states_grid;
	int i, j, cont;
	for (cont=0; cont<INIT_INFECTED; cont++) {
		i = random_uint (N, &seed);
		j = random_uint (N, &seed);
        if (grid[i][j] != I)
		    grid[i][j] = I;
        else
            cont--;
	}
}


void init_zero_vectors (COUNT_TYPE *isC, COUNT_TYPE *isH, COUNT_TYPE *isD, COUNT_TYPE *isI, COUNT_TYPE *isIa, COUNT_TYPE *isR) {
int i;
	for (i=0; i < TIME_STEPS+1; i++) {
		isC[i] = 0;
		isH[i] = 0;
		isD[i] = 0;
		isI[i] = 0;
		isIa[i] = 0;
		isR[i] = 0;
	}
}


int load_daily_confirmed_data (AC_TYPE *AC, int max_lenght) {	
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
	
	AC[0] = 0;
	fgets(str,max_len_comment,f);		
	while (!feof(f) && cont<max_lenght) {
		sscanf (str, "%c", &c);
		if (c != comment && c != '\n' && strlen(str)>0) {
			// If it doesn't begin with '#' or '\n' is not a comment or a newline.
			sscanf (str, "%d", &value);
			AC[cont] = value;
			cont++;
		}
		fgets(str,max_len_comment,f);
	}
	// Just in case there could be in the future longer vectors (periods of time).
	for (i=cont; i<max_lenght; i++) {
		AC[i] = 0;
	}
	return 0;
}


void init_pa_f3_RC (PROBABILITY_TYPE *pa_, PROBABILITY_TYPE *f3_, PROBABILITY_TYPE *RC_, CELL_TYPE *health_states_grid) {
	int i,j,ff1;
	float r1, r2, f1, f2;
	CELL_TYPE (*grid)[N] = (CELL_TYPE (*)[N]) health_states_grid;
	PROBABILITY_TYPE (*pa)[N] = (PROBABILITY_TYPE (*)[N]) pa_;
	PROBABILITY_TYPE (*f3)[N] = (PROBABILITY_TYPE (*)[N]) f3_;
	PROBABILITY_TYPE (*RC)[N] = (PROBABILITY_TYPE (*)[N]) RC_;
	
	ff1 = 1;
	for (i=LOWER_LIMIT; i<UPPER_LIMIT; i++) {
		for (j=LOWER_LIMIT; j<UPPER_LIMIT; j++) {
			if (grid[i][j] != E) {
				r1 = r4_uniform_01(&seed);
				// f1: probability of being male or female:
				if (r1 <= MALE_PROPORTION && r1>0 ) {
					f1 = Fmale;
				} else {
					f1 = Ffemale;
				}
				r2 = r4_uniform_01(&seed);
				// pa: probability of becoming asymptomatic:
				// f3: infectivity
				// RC: resitance/inmunity to the infection
				if (r2 <= AGE_0_4) {
					pa[i][j] = PA_0_4_YEARS;
					f3[i][j] = r4_uniform_01(&seed);
					f2 = Fage_below_60;
					RC[i][j] = ff1*f1*f2*r4_uniform_01(&seed);
				} else if (r2 <= AGE_5_14) {
					pa[i][j] = PA_5_14_YEARS;
					f3[i][j] = r4_uniform_01(&seed);
					f2 = Fage_below_60;
					RC[i][j] = ff1*f1*f2*r4_uniform_01(&seed);					
				} else if (r2 <= AGE_15_29) {
					pa[i][j] = PA_15_29_YEARS;
					f3[i][j] = r4_uniform_01(&seed);
					f2 = Fage_below_60;
					RC[i][j] = ff1*f1*f2*r4_uniform_01(&seed);					
				} else if (r2 <= AGE_30_59) {
					pa[i][j] = PA_30_59_YEARS;
					f3[i][j] = r4_uniform_01(&seed);
					f2 = Fage_below_60;
					RC[i][j] = ff1*f1*f2*r4_uniform_01(&seed);					
				} else if (r2 <= AGE_60_69) {
					pa[i][j] = PA_60_69_YEARS;
					f3[i][j] = r4_uniform_01(&seed);
					f2 = Fage_above_60;
					RC[i][j] = ff1*f1*f2*r4_uniform_01(&seed);					
				} else if (r2 <= AGE_70_79) {
					pa[i][j] = PA_70_79_YEARS;
					f3[i][j] = r4_uniform_01(&seed);
					f2 = Fage_above_60;
					RC[i][j] = ff1*f1*f2*r4_uniform_01(&seed);					
				} else {
					pa[i][j] = PA_80_YEARS;
					f3[i][j] = r4_uniform_01(&seed);
					f2 = Fage_above_60;
					RC[i][j] = ff1*f1*f2*r4_uniform_01(&seed);					
				}
			}
		}
	}
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
	
//void calculate_infection_transitions (CELL_TYPE *health_states_grid, PROBABILITY_TYPE *f3_, PROBABILITY_TYPE *RC_, PROBABILITY_TYPE *pa_) {
void calculate_infections (CELL_TYPE *health_states_grid, PROBABILITY_TYPE *pb_, PROBABILITY_TYPE *f3_, PROBABILITY_TYPE *RC_) {
	int i,j;
	float AA, BB;
	PROBABILITY_TYPE irc;	// inverse resistance/inmunity coefficient
	int ifx1, ifx2, ifx3, ifx4, ifx5, ifx6, ifx7, ifx8;
	int ifx11, ifx22, ifx33, ifx44, ifx55, ifx66, ifx77, ifx88;
	CELL_TYPE (*grid)[N] = (CELL_TYPE (*)[N]) health_states_grid;
	PROBABILITY_TYPE (*pb)[N] = (PROBABILITY_TYPE (*)[N]) pb_;
	PROBABILITY_TYPE (*f3)[N] = (PROBABILITY_TYPE (*)[N]) f3_;
	PROBABILITY_TYPE (*RC)[N] = (PROBABILITY_TYPE (*)[N]) RC_;

	for (i=LOWER_LIMIT; i<UPPER_LIMIT; i++) {
		for (j=LOWER_LIMIT; j<UPPER_LIMIT; j++) {
			if (grid[i][j] == S) {
				pb[i][j] = 0;
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
					pb[i][j] = AA*(0.5)/4.0 + BB*(1.0/(2*sqrt(2)))/4.0;
				} else {
					//pb[i][j] = 0;
				}
			}
		}
	}
}


void calculate_infection_transitions (CELL_TYPE *health_states_grid, PROBABILITY_TYPE *pb_, PROBABILITY_TYPE *pa_) {
	int i,j;
	CELL_TYPE (*grid)[N] = (CELL_TYPE (*)[N]) health_states_grid;
	PROBABILITY_TYPE (*pb)[N] = (PROBABILITY_TYPE (*)[N]) pb_;
	PROBABILITY_TYPE (*pa)[N] = (PROBABILITY_TYPE (*)[N]) pa_;
	
	for (i=LOWER_LIMIT; i<UPPER_LIMIT; i++) {
		for (j=LOWER_LIMIT; j<UPPER_LIMIT; j++) {
			if (grid[i][j] == S) {
				// The states of one individual in the cells are updated by following:
				if (pb[i][j] > r4_uniform_01(&seed) && pa[i][j] > r4_uniform_01(&seed)) {
					grid[i][j] = Ia;
				} else if (pb[i][j] > r4_uniform_01(&seed) && pa[i][j] <= r4_uniform_01(&seed)) {
					grid[i][j] = I;
				}
			}
		}
	}
}


void calculate_rest_state_transitions (CELL_TYPE *health_states_grid, CELL_TYPE *time_, int t, PROBABILITY_TYPE uu, PROBABILITY_TYPE kk, COUNT_TYPE *isC, COUNT_TYPE *isH, COUNT_TYPE *isD, COUNT_TYPE *isI, COUNT_TYPE *isIa, COUNT_TYPE *isR) {
int i,j;
	CELL_TYPE (*grid)[N] = (CELL_TYPE (*)[N]) health_states_grid;
	CELL_TYPE (*time)[N][N] = (CELL_TYPE (*)[N][N]) time_;
	// In Octave code 'time1' is now matrix time[0], 'time2' matrix is time[1], 
	// 'time3' is time[2], and 'time4' is time[3]. 
	
	for (i=LOWER_LIMIT; i<UPPER_LIMIT; i++) {
		for (j=LOWER_LIMIT; j<UPPER_LIMIT; j++) {
			if (grid[i][j] == Ia) {
				isIa[t] = isIa[t] + 1; 		// Current infected (asymptomatic) number in simulation
				time[3][i][j] = time[3][i][j] + 1;
			}
			
			if (time[3][i][j] == T1+T2) {
				grid[i][j] = R;
				time[3][i][j] = 0;
			}
			
			if (grid[i][j] == I) {
				isI[t] = isI[t] + 1; 		// Current infected (symptomatic) number in simulation
				time[0][i][j] = time[0][i][j] + 1;
			}
			
			if (time[0][i][j] == T1) {
				grid[i][j] = C;
				time[0][i][j] = 0;
				isC[t] = isC[t] + 1; 		// Daily confirmed in simulation
			}
			
			if (grid[i][j] == C) {
				time[1][i][j] = time[1][i][j] + 1;
			}

			
			if (time[1][i][j] == T2) {
				if (r4_uniform_01(&seed) > uu) {
					grid[i][j] = R;
				} else {
					grid[i][j] = H;
					isH[t] = isH[t] + 1;	// Daily hospitalized in simulation
				}
				time[1][i][j] = 0;
			}
				
			if (grid[i][j] == H) {
				time[2][i][j] = time[2][i][j] + 1;
			}
			
			if (time[2][i][j] == T3) {
				if (r4_uniform_01(&seed) > kk) {
					grid[i][j] = R;
				} else {
					grid[i][j] = D;
					isD[t] = isD[t] + 1;	// Daily dead in simulation
				}
				time[2][i][j] = 0;
			}
			
			if (grid[i][j] == R) {
				isR[t] = isR[t] + 1; 		// Cumulative recovered number in simulation
			}
		}
	}
}


void simulate_population_movements (CELL_TYPE *health_states_grid, PROBABILITY_TYPE *pa_, PROBABILITY_TYPE *f3_, PROBABILITY_TYPE *RC_, PROBABILITY_TYPE move_proportion, CELL_TYPE max_distance) {
	int i,j,di,dj;
	CELL_TYPE temp1;
	PROBABILITY_TYPE temp2;
	CELL_TYPE (*grid)[N] = (CELL_TYPE (*)[N]) health_states_grid;
	PROBABILITY_TYPE (*pa)[N] = (PROBABILITY_TYPE (*)[N]) pa_;
	PROBABILITY_TYPE (*f3)[N] = (PROBABILITY_TYPE (*)[N]) f3_;
	PROBABILITY_TYPE (*RC)[N] = (PROBABILITY_TYPE (*)[N]) RC_;

	for (i=LOWER_LIMIT+max_distance; i<UPPER_LIMIT-max_distance; i++) {
		for (j=LOWER_LIMIT+max_distance; j<UPPER_LIMIT-max_distance; j++) {
			if (r4_uniform_01 (&seed) > 1-move_proportion) { 
				// Swap 
				di = (1 + random_uint (max_distance, &seed)) - (1 + random_uint (max_distance, &seed));
				dj = (1 + random_uint (max_distance, &seed)) - (1 + random_uint (max_distance, &seed));
					// Do swapping
					temp1 = grid[i][j];
					grid[i][j] = grid[i+di][j+dj];
					grid[i+di][j+dj] = temp1;
										
					temp2 = pa[i][j];
					pa[i][j] = pa[i+di][j+dj];
					pa[i+di][j+dj] = temp2;

					temp2 = f3[i][j];
					f3[i][j] = f3[i+di][j+dj];
					f3[i+di][j+dj] = temp2;

					temp2 = RC[i][j];
					RC[i][j] = RC[i+di][j+dj];
					RC[i+di][j+dj] = temp2;					
			}
		}
	}
}


void calculate_self_isolation_state_transitions (CELL_TYPE *health_states_grid, AC_TYPE *AC, int t) {
	int i,j;
	PROBABILITY_TYPE q, qq;
	CELL_TYPE (*grid)[N] = (CELL_TYPE (*)[N]) health_states_grid;

	if (t>= 39) {
		q = 0.7 - 0.1*(AC[t-20]-AC[t-21])/AC[t-21]/0.025;
		for (i=LOWER_LIMIT; i<UPPER_LIMIT; i++) {
			for (j=LOWER_LIMIT; j<UPPER_LIMIT; j++) {
				// self-isolation proportion q 
				// correspond to Eq.(5)
				if (grid[i][j] == Si) {
					grid[i][j] = S;
				}
				qq = r4_uniform_01 (&seed);
				if (qq <= q && qq > 0 && grid[i][j] == S) {
					grid[i][j] = Si;
				}
			}
		}
	}
}


int write_results_to_file () {
	FILE *f;
	char filename[60];
	struct tm *timenow;
	int i;
	
	time_t now = time(NULL);
	timenow = gmtime(&now);
	strftime(filename, sizeof(filename), "covid19_serial_results_%Y-%m-%d_%H:%M:%S.txt", timenow);
	
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
		fprintf (f, "%6d %6d %6d %6d %6d %6d %6d \n", isC[i], isH[i], isD[i], isI[i], isIa[i], isR[i], isI[i]+isIa[i]);
	}

	return 0;
}


void free_main_simulation_variables () {
	if (AC != NULL)
		free (AC);				// Contain daily confirmed number in AC.txt
	if (grid != NULL)
		free (grid);			// Main state matrix 
	if (timer != NULL)
		free (timer);			// Timers matrix (time1, time2, time3)
	if (pb != NULL)		
		free (pb);				// Probability of being infected
	if (pa != NULL)		
		free (pa);				// Probability of becoming asymptomatic
	if (f3 != NULL)		
		free (f3);				// Infectivity of each person
	if (RC != NULL)		
		free (RC);				// Resistance / immunity to the infection
	
	if (isC != NULL)		
		free (isC);				// daily confirmed number in simulation
	if (isH != NULL)		
		free (isH);				// daily hospitalized number in simulation
	if (isD != NULL)		
		free (isD);				// daily dead number in simulation
	if (isI != NULL) 			
		free (isI); 			// current infected (symptomatic) number in simulation
	if (isIa != NULL)  	
		free (isIa);			// current infected (asymptomatic) number in simulation
	if (isR != NULL)  
		free (isR);				// cumulative recovered number in simulation

	#ifdef __VIDEO
		if (rgb != NULL)
			free (rgb);
	#endif
}



/******************************************************************************/
// Used for random numbers:
float r4_uniform_01 (unsigned int *seed )
	
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
	/*const int i4_huge = 2147483647;
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
	//r = ( float ) ( *seed ) * 4.656612875E-10;
	
	//return r;


   //return (float)rand() / (float)RAND_MAX;
   return (float)rand_r(seed) / (float)RAND_MAX;

}


/* Random integer in [0, limit).
   The best way to generate random numbers in C is to use a third-party library.
   The standard C function rand() is usually not recommended.
   OpenSSL's RAND_bytes() seeds itself. 
*/
unsigned int random_uint(unsigned int limit, unsigned int *seed) {
	/*union {
		unsigned int i;
		unsigned char c[sizeof(unsigned int)];
	} u;
	
	do {
		if (!RAND_bytes(u.c, sizeof(u.c))) {
			fprintf(stderr, "Can't get random bytes!\n");
			exit(1);
		}
	} while (u.i < (-limit % limit)); /* u.i < (2**size % limit) */
	//return u.i % limit;


   //return rand() % (limit);
   return rand_r(seed) % (limit);

}


void print_vector (AC_TYPE *v, int start, int end) {	
	// For Debug purposes
	int i;
	for (i=start; i<=end; i++) {
		printf ("%d ", v[i]);
	}
	printf ("\n");
}

void print_matrix (CELL_TYPE *m_, int start, int end) {	
	// For Debug purposes
	int i,j;
	CELL_TYPE (*m)[N] = (CELL_TYPE (*)[N]) m_;
	
	printf ("\n------------\n");
	for (i=start; i<=end; i++) {
		for (j=start; j<=end; j++) {
			printf ("%d ", m[i][j]);
		}
		printf ("\n");
	}
	printf ("------------\n\n");
}


// Functions used to generate CA evolution video:
#ifdef __VIDEO

// Creates a grid (health states) rgb screenshot.
uint8_t* generate_rgb (int width, int height, CELL_TYPE *grid_, uint8_t *rgb) {
	int x, y, cur;
	
	// Transform "grid" (CA states grid) into a similar rgb image format.
	CELL_TYPE (*grid)[height] = (CELL_TYPE (*)[height]) grid_;
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

