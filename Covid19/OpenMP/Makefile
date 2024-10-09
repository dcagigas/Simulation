CC=g++
CC2=g++
#CXXFLAGS= -O3 -Wextra -lm -lcrypto -lavcodec -lavformat -lavutil -lcrypto -lswscale

#CC_FLAGS= -O3 -Wextra -c -I/usr/include/opencv4 -fopenmp -lgomp
CC_FLAGS= -O3 -Wextra -c -fopenmp 

#CC2_FLAGS= -O3 -Wextra -std=c++11 -lm -lcrypto 
# Flags for the gprof profiler: -pg

#CC2_FLAGS= -O3 -Wextra -std=c++11 -lm -lcrypto -fopenmp -lgomp
CC2_FLAGS= -O3 -Wextra -std=c++11 -lm -fopenmp # -lcrypto 


OPENCV = `pkg-config opencv4 --cflags --libs` -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_imgcodecs -lopencv_videoio 


EXE=main

all: $(EXE)


covid19_NY_OpenMP.o: covid19_NY_OpenMP.c
	$(CC) $(CC_FLAGS) covid19_NY_OpenMP.c 

main: covid19_NY_OpenMP.o
	$(CC2) -o covid19_NY_OpenMP  covid19_NY_OpenMP.o $(CC2_FLAGS) 

clean:
	rm -rf covid19_NY_OpenMP *.o

