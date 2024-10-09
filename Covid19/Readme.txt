This source code or framework is associated with the following scientific paper:

Daniel Cagigas-Muñiz, Fernando Díaz-del-Río, José Luis Sevillano-Ramos, José-Luis Guisado-Lizar, "Parallelization Strategies for High-Performance and Energy-Efficient Epidemic Spread Simulations", submitted to the Simulation Modelling Practice and Theory journal in 2024.

This repository contains various implementations and results of a Covid-19 epidemic spread simulation model. The model simulates the evolution of the pandemic over 199 days in New York City (USA) in 2020. The simulation model is based on cellular automata, and the original Octave code was developed in the following paper:

Dai, Jindong; Zhai, Chi; Ai, Jiali; Ma, Jiaying; Wang, Jingde; Sun, Wei. 2021. "Modeling the Spread of Epidemics Based on Cellular Automata" Processes 9, no. 1: 55. https://doi.org/10.3390/pr9010055


This study explores alternative implementations aimed at improving not only speed-up but also energy efficiency. It is the first study to evaluate the energy consumption of parallel cellular automata simulations.

There are four implementations and their respective results:

- Octave: the original source code used to implement the Covid-19 model.

- Sequential C program: implementation of the Covid-19 model in C.

- OpenMP: parallel version of the sequential C program. Experiments were conducted using all available hardware threads on the tested multiprocessors (via Simultaneous Multi-threading) and also with only one hardware thread per multiprocessor core (without Simultaneous Multi-threading).

- CUDA: implementation of the Covid-19 model using a Graphics Processing Unit (GPU).

The source code is modular and parameterized, making it adaptable to study epidemic spread models beyond Covid-19. The implementations were developed and tested on the Linux Mint 21.1 operating system.


