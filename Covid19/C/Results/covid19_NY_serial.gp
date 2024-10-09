# USE GNUPLOT to obtain the .eps graphics.

set terminal postscript eps color

set xtics font "Helvetica,16"
set ytics font "Helvetica,16"
set xlabel "x-units" font ",18"
set ylabel "y-units" font ",18"
set xlabel "number of days"
set ylabel "number of confirmed people"
#set key tmargin
set key box inside
set key font ",18"
set key right top

set output "covid19_NY_serial_confirmed.eps"
#set title "Covid-19 evolution in New York from 6th March to 31st August 2020"
plot  'AC.txt' [1:180] [0:6000] using 1 title 'daily confirmed in actual data' with lines linewidth 2 lt rgb "#0000FF", 'covid19_serial_results.txt' [1:180] [0:6000] using 1 title 'daily confirmed in simulation' with lines linewidth 2 lt rgb "#FF0000"

set output "covid19_NY_serial_hospitalized.eps"
plot  'AC.txt' [1:180] [0:1600] using 2 title 'daily hospitalized in actual data' with lines linewidth 2 lt rgb "#0000FF", 'covid19_serial_results.txt' [1:180] [0:1600] using 2 title 'daily hospitalized in simulation' with lines linewidth 2 lt rgb "#FF0000"

set output "covid19_NY_serial_dead.eps"
plot  'AC.txt' [1:180] [0:600] using 3 title 'daily dead in actual data' with lines linewidth 2 lt rgb "#0000FF", 'covid19_serial_results.txt' [1:180] [0:600] using 3 title 'daily dead in simulation' with lines linewidth 2 lt rgb "#FF0000"

set ylabel "population"
set key right center
set output "covid19_NY_infected-cumulative_recovered.eps"
plot  'covid19_serial_results.txt' [1:178] using 7 title 'current infected' with lines linewidth 2 lt rgb "#000000",  'covid19_serial_results.txt' [1:178] using 5 title 'asymptomatic infected' with lines linewidth 2 lt rgb "#FF0000", 'covid19_serial_results.txt' [1:178] using 6 title 'cummulative recovered' with lines linewidth 2 lt rgb "#00FF00",


reset
# pause -1 "Press 'Return' to continue"
