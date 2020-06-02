clear all
close all
%% Figure dimensions
figure(1)
set(gcf,'PaperUnits','centimeters')
%Setting the units of the figure on paper to centimeters.
xSize = 13; ySize = 13;
%Size of the figure
xLeft = (21-xSize)/2; yTop = (30-ySize)/2;
%Coordinates to center the figure on A4-paper
set(gcf,'PaperPosition',[xLeft yTop xSize ySize])
%This command sets the position and size of the figure on the paper to the desired values.
set(gcf,'Position',[0.5 0.5 xSize*50 ySize*50])
set(gcf, 'Color', 'w');

%% Parameter Initialization:
duration=5000; % duration in ms
dt=0.1; % simulation time step.
tau=50; % Filter time for the input.
tRef=5; % Refractory period for the spike trains.
nn=100; % Number of spiketrains we seek to create later.
spikebin=5; % Number of ms per PSTH bin.
Timevector=(0.1:dt:duration);
% A vector of time in ms, in steps of dt.
WhiteNoise=rand(1,length(Timevector))-0.5;
% uniform white noise drawn from +/- 0.5 as long as the time vector.
FilteredWhiteNoise=WhiteNoise.*0;
% an empty vector which we will use to create the time-filtered input.
SpikeTrains=zeros(nn,length(Timevector));
%A Matrix that will hold all spiketrains.
PloTrains=SpikeTrains;
% This is just a plotting variable to oercome a screen resolution problem in matlab.
avetrain=0;
% A counter to calculate the average firing rate.
tslt=0;
% (== t(ime)s(ince)l(ast)(t)oggle (this serves as a Boolean for the sparsification of the input signal

tsls=zeros(nn,1);
% (== t(ime)s(ince)l(ast)(s)pike (to keep track of the refractory period of each spike train)
BinnedSpikeTrains=zeros(1,duration/spikebin);
% a vector to create a PSTH with binwidth “spikebin” from the spike trains.


%% Making the time-filtered white noise signal:
for t=2:duration/dt
FilteredWhiteNoise(t) = WhiteNoise (t) - ...
(WhiteNoise (t) - FilteredWhiteNoise(t-1))*exp(-dt/tau);
end
%% This routine changes the signal trace ”FilteredWhiteNoise” by a ”exp(-dt/tau)” fraction of
% the difference between the signal and a random number.
FilteredWhiteNoise=FilteredWhiteNoise./max(FilteredWhiteNoise);
%Normalize to a maximum value of 1.

%% Plotting:
figure(1)
subplot(4,1,1)
plot(Timevector, FilteredWhiteNoise)
axis([0 duration -1 1])
x=sprintf('Time Filtered White Noise (FWN)');
title (x)
%% Normalize and Rectify:
FilteredWhiteNoise=FilteredWhiteNoise.*(500*dt/1000);
% Normalizes the trace to a peak value of 500Hz*dt (=0.05).
FilteredWhiteNoise(FilteredWhiteNoise<0)=0;
%Sets all negative values of ”FilteredWhiteNoise” to 0.
%% Plotting:
subplot(4,1,2)
hold on
plot(Timevector,FilteredWhiteNoise, 'b', 'LineWidth', 1.1)
%% Sparsifieing the Input Signals:
% This routine goes through the signal trace and deletes entire ”activity bumps” if certain conditions are fullfilled:
toggle = 0;
tslt=0;
for d=1:duration/dt-1
    % Routine becomes active (sets toggle == 1) if the signal is ==0, and the toggle is off (==0) and has been off for at least 1 ms:
    if(FilteredWhiteNoise(d)==0 && toggle==0 && (d-tslt>10))
        toggle=1; % toggle set
        tslt=d; % ”refractory” for toggle is set
    end
    % If routine active, every signal value is set to zero:
    if (toggle==1)
        FilteredWhiteNoise(d) = 0;
        % If routine has been active for longer than 0.5 ms, and the signal is 0, routine becomes inactive:
        if (FilteredWhiteNoise(d+1)==0 && (d-tslt>5))
            toggle=0;
        end
    end
end
%% Plotting:
subplot(4,1,2)
hold on
plot(Timevector, FilteredWhiteNoise, 'r')
axis([0 duration -0.005 0.05])
title ('Rectified & callibrated (blue) and sparsened (red) FWN')
%% Adding background firing rate:
FilteredWhiteNoise=FilteredWhiteNoise+(5*dt/1000);
% This is adjusted so that without any FilteredWhiteNoise the firing rate is 5 Hz*dt (0.0005).
%% Creating 100 spike trains:
for i=1:nn
    for t=1:duration/dt
        if (tsls (i) <= 0) % Allows potential spike if refractory period has subsided
            if(rand<FilteredWhiteNoise(t))
                SpikeTrains (i,t) = 1;%Fires if random variable < “FilteredWhiteNoise”.
                tsls (i) = tRef;% Sets the absolute refractory period.
                avetrain=avetrain+1;% Counts the total number of spikes.
                if(duration/dt-t>25)% This is just a plotting routine.
                    PloTrains (i,t:t+25)=1;%(Spike is elongated for plotting.)
                end
            end
        else
            tsls (i)=tsls (i) -dt;% subtracts dt from refractory counter if it is still >0.
        end
    end
end

avetrain=avetrain/(nn*duration/1000); %Calculates the average firing rate in Hz

%% Plotting:
subplot(4,1,3)
imagesc(-PloTrains)
colormap(gray)
title ('100 Spiketrains')

%% Recording a PSTH / Binned Input rate:
% This bins the spikes into bins and calculates the instantaneous firing rate in Hz.
for i=1:(duration/spikebin)-1
    BinnedSpikeTrains(i)=...
    sum(sum(SpikeTrains(:,((i-1)*(spikebin/dt))+1:(i*(spikebin/dt)))));
end
BinnedSpikeTrains= (BinnedSpikeTrains*(1000/spikebin))/nn;

%% Plotting:
subplot(4,1,4)
plot(BinnedSpikeTrains)
x=sprintf('Average input rate for 1 excitatory channel, %3.2f Hz, peak %3.2f Hz', avetrain, max(BinnedSpikeTrains));
title (x)