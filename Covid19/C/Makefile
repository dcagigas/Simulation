CC=g++
CC2=g++
CC_FLAGS= -O3 -Wextra -c -I/usr/include/opencv4  # -pg
#CC2_FLAGS= -O3 -Wextra -std=c++11 -lm -lcrypto 
# Flags for the gprof profiler:
CC2_FLAGS= -O3 -Wextra -std=c++11 -lm  # -pg #-lcrypto 
OPENCV = `pkg-config opencv4 --cflags --libs` -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_imgcodecs -lopencv_videoio 


EXE=main

all: $(EXE)


covid19_NY_serial.o: covid19_NY_serial.c
	$(CC) $(CC_FLAGS) covid19_NY_serial.c 

main: covid19_NY_serial.o
	$(CC2) -o covid19_NY_serial  covid19_NY_serial.o $(CC2_FLAGS) 
#	$(CC2) -o covid19_NY_serial  covid19_NY_serial.o $(CC2_FLAGS) $(OPENCV)

clean:
	rm -rf covid19_NY_serial   *.o

