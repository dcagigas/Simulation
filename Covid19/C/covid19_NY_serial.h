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

Copyright 2021 Daniel Cagigas-Muñiz (dcagigas@us.es)

	Also: 	Copyright (c) 2001 Fabrice Bellard for the video processing part
			available at https://github.com/FFmpeg/FFmpeg/blob/master/doc/examples/encode_video.c
			Copyright (c) 2004 John Burkardt for the normal distribution part
			available at https://people.sc.fsu.edu/~jburkardt/c_src/normal/normal.html
*/

//#define __VIDEO	// Uncomment if a Celular Automaton video is generated

/*
// Covid19 CA states:
#define S   0   // Un-infected
#define E   1   // Empty cell
#define Si  2   // Self-isolated
#define I   3   // Infected
#define C   4   // Confirmed
#define R   5   // Recovered
#define H   6   // Hospitalized
#define D   7   // Dead
#define Ia  8   // Asymptomatic

// Constants related to inmunity
#define Fmale           0.8059
#define Ffemale         1.0
#define Fage_above_60   0.7673
#define Fage_below_60   1.0

// Constants related to hospitalization fraction (u)
#define U_6_MARCH_TO_23_APRIL   0.31
#define U_24_APRIL_TO_12_JUNE   0.18
#define U_13_JUNE_TO_1_AUGUST   0.12
#define U_AFTER_1_AUGUST        0.11

// Constants related to dead fraction (k)
#define K_6_MARCH_TO_23_APRIL   0.38
#define K_24_APRIL_TO_12_JUNE   0.33
#define K_13_JUNE_TO_1_AUGUST   0.42
#define K_AFTER_1_AUGUST        0.17

// Constants related to the probability of becoming asymptomatic for different ages
#define PA_0_4_YEARS        0.95
#define PA_5_14_YEARS       0.8
#define PA_15_29_YEARS      0.7
#define PA_30_59_YEARS      0.5
#define PA_60_69_YEARS      0.4
#define PA_70_79_YEARS      0.3
#define PA_80_YEARS         0.2

// Constants related to age population proportion
#define AGE_0_4     0.06    // Age 0-4      is 6%
#define AGE_5_14    0.18    // Age 5-14     is 12%
#define AGE_15_29   0.41    // Age 15-29    is 23%
#define AGE_30_59   0.83    // Age 30-59    is 42%
#define AGE_60_69   0.92    // Age 60-69    is 9%
#define AGE_70_79   0.97    // Age 70-79    is 5%
                            // Age above 80 is 3%

// Male/female proportion
#define MALE_PROPORTION     0.477   // Population male proportion is 47.7%

// Constants related to the simulation
#define N               1001    	// CA grid (region) size 
#define TIME_STEPS		199			// Simulated days
#define INIT_INFECTED   250     	// Number of initial infected
#define TIME_STEPS      199     	// 199 days of 2020 year are simulated
#define T1              10      	// Period from infected (I) to confirmed (C)
#define T2              4       	// Period from confirmed (C) to hospitalized (H)
#define T3              4       	// Period from hospitalized (H) to recovered (R)
#define VACANCY_RATIO   0.2     	// Empty cell perdentage in the CA
#define L               10      	// Maximun distance cell movement
#define MOVE_PROPORTION 0.16    	// (cells) Moving proportion at each time step
#define DAILY_CONFIRMED_FILE "AC.txt"	// This file contains daily confirmed infections.
#define AC_DATA_LENGHT	180			// The "AC.txt" file contains 179 day data

#define LOWER_LIMIT 2				// 0 and 1 ghost rows/columns are not computed
#define UPPER_LIMIT N-2				// N-1 and N-2 ghost rows/columns are not computed

// Constants related to the video generation.
#define FPS 1 						// Frames per second
#define CODEC_NAME "libx264rgb"		// Video codec that support RGB format
*/

// Data types.
typedef  unsigned char CELL_TYPE;   // Used for CA grid states and timers (1,2,3,4)
typedef  int AC_TYPE;      			// Used for AC daily confirmed number 
typedef  unsigned int COUNT_TYPE;	// Used for recording statistics (isC, isH, isD, etc)
typedef  float PROBABILITY_TYPE;    // Used for Probability values (pa, f3, RC, etc)


// Covid19 CA states:
const CELL_TYPE S = 0;   // Un-infected
const CELL_TYPE E = 1;   // Empty cell
const CELL_TYPE Si = 2;   // Self-isolated
const CELL_TYPE I = 3;   // Infected
const CELL_TYPE C = 4;   // Confirmed
const CELL_TYPE R = 5;   // Recovered
const CELL_TYPE H = 6;   // Hospitalized
const CELL_TYPE D = 7;   // Dead
const CELL_TYPE Ia = 8;   // Asymptomatic

// Constants related to inmunity
const PROBABILITY_TYPE Fmale     =      0.8059;
const PROBABILITY_TYPE Ffemale   =      1.0;
const PROBABILITY_TYPE Fage_above_60 =  0.7673;
const PROBABILITY_TYPE Fage_below_60 =  1.0;

// Constants related to hospitalization fraction (u)
const PROBABILITY_TYPE U_6_MARCH_TO_23_APRIL =  0.31;
const PROBABILITY_TYPE U_24_APRIL_TO_12_JUNE =  0.18;
const PROBABILITY_TYPE U_13_JUNE_TO_1_AUGUST =  0.12;
const PROBABILITY_TYPE U_AFTER_1_AUGUST     =   0.11;

// Constants related to dead fraction (k)
const PROBABILITY_TYPE K_6_MARCH_TO_23_APRIL =  0.38;
const PROBABILITY_TYPE K_24_APRIL_TO_12_JUNE =  0.33;
const PROBABILITY_TYPE K_13_JUNE_TO_1_AUGUST =  0.42;
const PROBABILITY_TYPE K_AFTER_1_AUGUST     =   0.17;

// Constants related to the probability of becoming asymptomatic for different ages
const PROBABILITY_TYPE PA_0_4_YEARS     =   0.95;
const PROBABILITY_TYPE PA_5_14_YEARS    =   0.8;
const PROBABILITY_TYPE PA_15_29_YEARS   =   0.7;
const PROBABILITY_TYPE PA_30_59_YEARS   =   0.5;
const PROBABILITY_TYPE PA_60_69_YEARS   =   0.4;
const PROBABILITY_TYPE PA_70_79_YEARS   =   0.3;
const PROBABILITY_TYPE PA_80_YEARS      =   0.2;

// Constants related to age population proportion
const PROBABILITY_TYPE AGE_0_4   =  0.06;    // Age 0-4      is 6%
const PROBABILITY_TYPE AGE_5_14  =  0.18;    // Age 5-14     is 12%
const PROBABILITY_TYPE AGE_15_29 =  0.41;    // Age 15-29    is 23%
const PROBABILITY_TYPE AGE_30_59 =  0.83;    // Age 30-59    is 42%
const PROBABILITY_TYPE AGE_60_69 =  0.92;    // Age 60-69    is 9%
const PROBABILITY_TYPE AGE_70_79 =  0.97;    // Age 70-79    is 5%
// Age above 80 is 3%

// Male/female proportion
const PROBABILITY_TYPE MALE_PROPORTION  =   0.477;   // Population male proportion is 47.7%

// Constants related to the simulation
const COUNT_TYPE N  =             1001;    // CA grid (region) size 
const COUNT_TYPE TIME_STEPS	=	199;		// Simulated days
const COUNT_TYPE INIT_INFECTED =  250;     // Number of initial infected
const COUNT_TYPE T1           =   10;      // Period from infected (I) to confirmed (C)
const COUNT_TYPE T2           =   4;       // Period from confirmed (C) to hospitalized (H)
const COUNT_TYPE T3           =   4;       // Period from hospitalized (H) to recovered (R)
const PROBABILITY_TYPE VACANCY_RATIO =  0.2;     // Empty cell perdentage in the CA
const COUNT_TYPE L            =   10;      // Maximun distance cell movement
const PROBABILITY_TYPE MOVE_PROPORTION = 0.16;    // (cells) Moving proportion at each time step
//const std::string DAILY_CONFIRMED_FILE = "AC.txt";	// This file contains daily confirmed infections.
const COUNT_TYPE AC_DATA_LENGHT	= 180;		// The "AC.txt" file contains 179 day data


const COUNT_TYPE LOWER_LIMIT = 2;			// Ghost rows/columns are not treated
const COUNT_TYPE UPPER_LIMIT = N-2;
