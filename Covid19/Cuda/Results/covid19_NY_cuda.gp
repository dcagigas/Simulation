# USE GNUPLOT to obtain the .eps graphics.

set terminal postscript eps color

set xtics font "Helvetica,21"
set ytics font "Helvetica,21"

set xlabel "x-units" font ",26"
set ylabel "y-units" font ",26"
set xlabel "number of days"
set ylabel "number of confirmed people"

set xlabel offset 0,-1
set ylabel off -2.5,0
set lmargin 12
set bmargin 5

#set key tmargin
#set key box inside
set key font ",25"
set key right top

set output "covid19_NY_cuda_confirmed_b.eps"
plot  'AC.txt' [1:180] [0:6000] using 1 title 'daily confirmed in actual data' with lines linewidth 2 lt rgb "#0000FF", 'covid19_cuda_results.txt' [1:180] [0:6000] using 1 title 'daily confirmed in simulation' with lines linewidth 2 lt rgb "#FF0000"

set output "covid19_NY_cuda_hospitalized_b.eps"
plot  'AC.txt' [1:180] [0:1600] using 2 title 'daily hospitalized in actual data' with lines linewidth 2 lt rgb "#0000FF", 'covid19_cuda_results.txt' [1:180] [0:1600] using 2 title 'daily hospitalized in simulation' with lines linewidth 2 lt rgb "#FF0000"

set output "covid19_NY_cuda_dead_b.eps"
plot  'AC.txt' [1:180] [0:600] using 3 title 'daily dead in actual data' with lines linewidth 2 lt rgb "#0000FF", 'covid19_cuda_results.txt' [1:180] [0:600] using 3 title 'daily dead in simulation' with lines linewidth 2 lt rgb "#FF0000"

set ylabel "population (in thousands)"
set key right center
set format y "%.0s%c"
#set format y "%.0s"
set output "covid19_NY_infected-cumulative_recovered_b.eps"
plot  'covid19_cuda_results.txt' [1:178] using 7 title 'current infected' with lines linewidth 2 lt rgb "#000000",  'covid19_cuda_results.txt' [1:178] using 5 title 'asymptomatic infected' with lines linewidth 2 lt rgb "#FF0000", 'covid19_cuda_results.txt' [1:178] using 6 title 'cummulative recovered' with lines linewidth 2 lt rgb "#00FF00",


reset
# pause -1 "Press 'Return' to continue"
