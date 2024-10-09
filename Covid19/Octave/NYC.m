% Original code from:
% Dai, Jindong; Zhai, Chi; Ai, Jiali; Ma, Jiaying; Wang, Jingde; Sun, Wei. 2021. 
% "Modeling the Spread of Epidemics Based on Cellular Automata" Processes 9, no. 1: 55. https://doi.org/10.3390/pr9010055

% before run this file, please load the AC.mat first, 
% AC.mat is the actucal data of New York City in terms of daily confirmed,hospitalized and dead number (7 day moving average)
% AC.mat is ranged from March 6 to August 31
% this file is corresponding to Figure 3 in the paper

% This added to the original code to load "AC.mat".
% The "AC.mat" has been converted in a .csv file previously.

AC=dlmread("AC.csv",",")


n = 1001; %region size

X=zeros(n,n);%0 for S
X(n-7,n)=8;  %8 for Ia
X(n,n)=7;    %7 for D
X(n-1,n)=6;  %6 for H
X(n-2,n)=5;  %5 for R
X(n-3,n)=4;  %4 for C
X(n-4,n)=3;  %3 for I
X(n-5,n)=2;  %2 for Si
X(n-6,n)=1;  %1 for N

isC = 0;  % daily confirmed number in simulation
isH = 0;  % daily hospitalized number in simulation
isD = 0;  % daily dead number in simulation

% set for vacancy ratio = 0.2 
K1 = zeros(n,n); 
for i = 3:n-2
    for j = 3:n-2
        K1(i,j) = rand(1,1);
        if K1(i,j) <= 0.2  && K1(i,j) > 0  
            X(i,j) = 1;
        end
    end
end

% initial infected number
init = 250;  
X(randperm(numel(X),init)) = 3; 

T1=10; % period from infected to confirmed
T2=4;  % period from confirmed to hospitalized
T3=4;  % period from hospitalized to recovered

% pa and f are corresponding to Table A3 and Table A4
pa = zeros(n,n);  % The probability of becoming asymptomatic patients of different age groups
r1 = zeros(n,n); 
f1 = zeros(n,n);  % immunity coefficient of male and female individuals
for i = 3:n-2
    for j = 3:n-2
        if X(i,j) ~= 1
        r1(i,j) = rand(1,1);
        end
    end
end

for i = 3:n-2
    for j = 3:n-2
        if r1(i,j) <= 0.477  && r1(i,j) > 0   % male proportion is 47.7%
            f1(i,j) = 0.8059; % immunity coefficient of male individuals
        else f1(i,j) = 1;     % immunity coefficient of female individuals
        end
    end
end

r2 = zeros(n,n); 
f2 = zeros(n,n);  % immunity coefficient of young and old individuals
for i = 3:n-2
    for j = 3:n-2
        if X(i,j) ~= 1
        r2(i,j) = rand(1,1);
        end
    end
end

for i = 3:n-2
    for j = 3:n-2
        r2(i,j) = rand(1,1);
        if r2(i,j) <= 0.06;  %0-4
            pa(i,j) = 0.95; % The probability of becoming asymptomatic patients of different age groups
            f2(i,j) = 1; % immunity coefficient of young individuals
        else if r2(i,j) > 0.06 && r2(i,j) <= 0.18;  %5-14
                pa(i,j) = 0.8;
                f2(i,j) = 1; % immunity coefficient of young individuals
            else if r2(i,j) > 0.18 && r2(i,j) <= 0.41;  %15-29
                    pa(i,j) = 0.7;
                    f2(i,j) = 1; % immunity coefficient of young individuals
                else if r2(i,j) > 0.41 && r2(i,j) <= 0.83;  %30-59
                        pa(i,j) = 0.5;
                        f2(i,j) = 1; % immunity coefficient of young individuals
                    else if r2(i,j) > 0.83 && r2(i,j) <= 0.92;  %60-69
                            pa(i,j) = 0.4;
                            f2(i,j) = 0.7673;  % immunity coefficient of old individuals
                        else if r2(i,j) > 0.92 && r2(i,j) <= 0.97;  %70-79
                                pa(i,j) = 0.3;
                                f2(i,j) = 0.7673;  % immunity coefficient of old individuals
                            else if r2(i,j) > 0.97;  %80-
                                    pa(i,j) = 0.2;
                                    f2(i,j) = 0.7673;  % immunity coefficient of old individuals
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

 
f3 = zeros(n,n);  %  infectivity
for i = 3:n-2
    for j = 3:n-2
        if X(i,j) ~= 1
        f3(i,j) = rand(1,1);
        end
    end
end

ff1 = 1; 
RC = zeros(n,n);  % resistance / immunity to the infection
for i = 3:n-2
    for j = 3:n-2
        if X(i,j) ~= 1
        RC(i,j) = ff1*f1(i,j)*f2(i,j)*rand(1,1);
        end
    end
end

time1 = zeros(n,n); % record infected time of symptomatic ones
time2 = zeros(n,n); % record the time after confirmed
time3 = zeros(n,n); % record hospitalized time
time4 = zeros(n,n); % record infected time of asymptomatic ones

aa = 1/2;
bb = 1/(2*2^0.5);

dd = 10; % maximum moving step length L

for t = 1:199
    disp(t)
    % The set value of hospitalization fraction (u) and dead fraction (k)
    % correspond to Table A5
   if t <= 70
        uu = 0.31;
        kk = 0.38;
    else if t <= 120
            uu = 0.18;
            kk = 0.33;
        else if t <= 170
                uu = 0.12;
                kk = 0.42;
                else  
                uu = 0.11;
                kk = 0.17;
            end
        end
   end
   
   % Possibility of being infected for each individual
   % correspond to Eq.(3)
    P = zeros(n,n);
    for i = 3:n-2
        for j = 3:n-2
            if X(i,j) == 0 && (X(i-1,j-1) == 3 ||  X(i-1,j) == 3 ||  X(i-1,j+1) == 3||  X(i,j-1) == 3||  X(i,j+1) == 3||  X(i+1,j-1) == 3 ||  X(i+1,j) == 3 ||  X(i+1,j+1) == 3 || X(i-1,j-1) == 8 ||  X(i-1,j) == 8 ||  X(i-1,j+1) == 8||  X(i,j-1) == 8||  X(i,j+1) == 8||  X(i+1,j-1) == 8 ||  X(i+1,j) == 8 ||  X(i+1,j+1) == 8)
                ifx1 = (X(i-1,j-1) == 3);
                ifx2 = (X(i-1,j) == 3);
                ifx3 = (X(i-1,j+1) == 3);
                ifx4 = (X(i,j-1) == 3);
                ifx5 = (X(i,j+1) == 3);
                ifx6 = (X(i+1,j-1) == 3);
                ifx7 = (X(i+1,j) == 3);
                ifx8 = (X(i+1,j+1) == 3);
                ifx11 = (X(i-1,j-1) == 8);
                ifx22 = (X(i-1,j) == 8);
                ifx33 = (X(i-1,j+1) == 8);
                ifx44 = ( X(i,j-1) == 8);
                ifx55 = (X(i,j+1) == 8);
                ifx66 = (X(i+1,j-1) == 8);
                ifx77 = (X(i+1,j) == 8);
                ifx88 = (X(i+1,j+1) == 8);
                AA = (f3(i-1,j)*(ifx2)*(1-RC(i,j)))^0.5 + (f3(i,j+1)*(ifx5)*(1-RC(i,j)))^0.5+ (f3(i+1,j)*(ifx7)*(1-RC(i,j)))^0.5+ (f3(i,j-1)*(ifx4)*(1-RC(i,j)))^0.5 + (0.5*f3(i-1,j)*(ifx22)*(1-RC(i,j)))^0.5 + (0.5*f3(i,j+1)*(ifx55)*(1-RC(i,j)))^0.5+ (0.5*f3(i+1,j)*(ifx77)*(1-RC(i,j)))^0.5+ (0.5*f3(i,j-1)*(ifx44)*(1-RC(i,j)))^0.5;
                BB = (f3(i-1,j-1)*(ifx1)*(1-RC(i,j)))^0.5 + (f3(i-1,j+1)*(ifx3)*(1-RC(i,j)))^0.5+ (f3(i+1,j-1)*(ifx6)*(1-RC(i,j)))^0.5+ (f3(i+1,j+1)*(ifx8)*(1-RC(i,j)))^0.5 + (0.5*f3(i-1,j-1)*(ifx11)*(1-RC(i,j)))^0.5 + (0.5*f3(i-1,j+1)*(ifx33)*(1-RC(i,j)))^0.5+ (0.5*f3(i+1,j-1)*(ifx66)*(1-RC(i,j)))^0.5+ (0.5*f3(i+1,j+1)*(ifx88)*(1-RC(i,j)))^0.5;
                P(i,j) = AA *aa /4 + BB *bb /4 ;
            end
        end
    end
    
    %The states of one individual in the cells are updated by following:
    for i = 3:n-2
        for j = 3:n-2
            if X(i,j) == 0 && P(i,j) > rand(1,1) && pa(i,j) > rand(1,1)
                X(i,j) = 8;
            else if X(i,j) == 0 && P(i,j) > rand(1,1) && pa(i,j) <= rand(1,1)
                    X(i,j) = 3;
                end
            end
        end
    end
    
    for i = 3:n-2
        for j = 3:n-2
            if X(i,j) == 8
                time4(i,j) = time4(i,j) +1;
            end
        end
    end
    
    for i = 3:n-2
        for j = 3:n-2
            if time4(i,j) == T1 + T2
                X(i,j) = 5;
                time4(i,j) = 0;
            end
        end
    end
    
    for i = 3:n-2
        for j = 3:n-2
            if X(i,j) == 3
                time1(i,j) = time1(i,j) +1;
            end
        end
    end
    
    isC = sum(sum(time1 == T1));  % daily confirmed in simulation
    
    for i = 3:n-2
        for j = 3:n-2
            
            if time1(i,j) == T1
                X(i,j) = 4;
                time1(i,j) = 0;
            end
            
            if X(i,j) == 4
                time2(i,j) = time2(i,j) + 1;
            end
        end
    end
    
    isH = round(uu * sum(sum(time2 == T2)));   % daily hospitalized in simulation

    for i = 3:n-2
        for j = 3:n-2
            if time2(i,j) == T2
                if rand(1,1) > uu
                    X(i,j) = 5;
                    time2(i,j) = 0;
                else X(i,j) = 6;
                    time2(i,j) = 0;
                end
            end
            
            if X(i,j) == 6
                time3(i,j) =time3(i,j) + 1;
            end
        end
    end

    
    isD = round(kk * sum(sum(time3 == T3)));   % daily dead in simulation
    
    for i = 3:n-2
        for j = 3:n-2
            if time3(i,j) == T3
                if rand(1,1) > kk
                    X(i,j) = 5;
                    time3(i,j) = 0;
                else X(i,j) = 7;
                    time3(i,j) = 0;
                end
            end
        end
    end
       
  % moving proportion m=0.16
   mov = 0.16;
   m = zeros(n,n);
   ss = 0; 
   for i = 13:n-12
        for j = 13:n-12
            m(i,j) = rand(1,1);
            if m(i,j) > 1-mov     
                ss = ss + 1;
            end
        end
   end
            
    rr1 = zeros(n,n); % record X 
    rr2 = zeros(n,n); % record f3 
    rr3 = zeros(n,n); % record \BC\C7¼RC 
    rr4 = zeros(n,n); % record pa 


    d1 = zeros(n,n);
    d2 = zeros(n,n);
    
    % realize the movement (m=0.16,d=10),0.16 of the cell move within 10
    for i = 13:n-12
        for j = 13:n-12
            if m(i,j) > 1-mov
                d1(i,j) = randperm(dd,1) -randperm(dd,1);
                d2(i,j) = randperm(dd,1) -randperm(dd,1);
                
                rr1(i,j) = X(i,j);
                X(i,j) = X(i+d1(i,j),j+d2(i,j));
                X(i+d1(i,j),j+d2(i,j)) = rr1(i,j);
                
                rr2(i,j) = f3(i,j);
                f3(i,j) = f3(i+d1(i,j),j+d2(i,j));
                f3(i+d1(i,j),j+d2(i,j)) = rr2(i,j);
                
                rr3(i,j) = RC(i,j);
                RC(i,j) = RC(i+d1(i,j),j+d2(i,j));
                RC(i+d1(i,j),j+d2(i,j)) = rr3(i,j);
                
                rr4(i,j) = pa(i,j);
                pa(i,j) = pa(i+d1(i,j),j+d2(i,j));
                pa(i+d1(i,j),j+d2(i,j)) = rr4(i,j);
                
            end
        end
    end
    
    % self-isolation\A3\ACS \A1\FA Si
    for i = 3:n-2
        for j = 3:n-2
            if X(i,j) == 2
               X(i,j) = 0;
            end
        end
    end
    
    % self-isolation proportion q   
    qq = zeros(n,n);
    q = 0;   
    % correspond to Eq.(5)
    if t >= 39
    q = 0.7 - 0.1*(AC(t-20,1)-AC(t-21,1))/AC(t-21,1)/0.025;
    end
    
        for i = 3:n-2
            for j = 3:n-2
                qq(i,j) = rand(1,1);
                if qq(i,j) > 0 && qq(i,j) <= q && X(i,j) ==0
                    X(i,j) = 2;
                end
            end
        end
    
isI = sum(sum(X==3))-1;   % current infected (symptomatic) number in simulation
isIa = sum(sum(X==8))-1;  % current infected (asymptomatic) number in simulation
isR = sum(sum(X==5))-1;   % culmulative recovered number in simulation

W(t,:) = [isC(:) isH(:) isD(:) isI(:) isIa(:) isR(:)  isI(:)+isIa(:)];

end

save octave_out.txt W

%draw the figures (Figure 3 in the paper)
figure(2)
plot(1:178,AC(1:178,1),'b','LineWidth',3) 
hold on 
plot(1:178,W(22:199,1),'r','LineWidth',3) 
xlabel('number of days','Fontname', 'Times New Roman','fontsize',16);
ylabel('number of confirmed people','Fontname', 'Times New Roman','fontsize',16);
legend('daily confirmed in actual data','daily confirmed in simulation')
set(legend,'Fontname','Times New Roman','FontSize',16)
print -f2 figure2.pdf

figure(3)
plot(1:178,AC(1:178,2),'b','LineWidth',3) 
hold on 
plot(1:178,W(22:199,2),'r','LineWidth',3) 
xlabel('number of days','Fontname', 'Times New Roman','fontsize',16);
ylabel('number of hospitalized people','Fontname', 'Times New Roman','fontsize',16);
legend('daily hospitalized in actual data','daily hospitalized in simulation')
set(legend,'Fontname','Times New Roman','FontSize',16)
print -f3 figure3.pdf

figure(4)
plot(1:178,AC(1:178,3),'b','LineWidth',3) 
hold on 
plot(1:178,W(22:199,3),'r','LineWidth',3) 
xlabel('number of days','Fontname', 'Times New Roman','fontsize',16);
ylabel('number of dead people','Fontname', 'Times New Roman','fontsize',16);
legend('daily dead in actual data','daily dead in simulation')
set(legend,'Fontname','Times New Roman','FontSize',16)
print -f4 figure4.pdf

figure(5)
plot(1:178,W(22:199,7),'k','LineWidth',3) 
hold on 
plot(1:178,W(22:199,5),'m','LineWidth',3) 
hold on 
plot(1:178,W(22:199,6),'g','LineWidth',3) 
xlabel('number of days','Fontname', 'Times New Roman','fontsize',16);
ylabel('population','Fontname', 'Times New Roman','fontsize',16);
legend('current infected','asymptomatic infected','cumulative recovered')
set(legend,'Fontname','Times New Roman','FontSize',16)
print -f5 figure5.pdf

xx = corrcoef(W(22:199,1),AC(1:178,1));
yy = corrcoef(W(22:199,2),AC(1:178,2));
zz = corrcoef(W(22:199,3),AC(1:178,3));

quit

