% AVERAGING THE 3 SUBJECT TRIALS

clc
clear

fs=250;

% Specify sessions to average
session_names = {'18_02_24_rocio3', '18_02_24_rocio3', '18_02_24_rocio3'};

% Use averaging function
[normal, deviant, assr, assr_f, ec, eo, ec_sd, eo_sd, spect, ssvep, ssvep_f] = average_trials(session_names);

%% P300

% Parameters
eegsize = size(normal);
timeax = 0:1/fs:(eegsize(1,2)-1)/fs;

% Plot the average deviant and non-deviant responses for each channel
f1= figure(1);
set(f1, 'color','w')

for i=1:4
    subplot(1,4,i)
    hold all
    grid on
    xlabel('Time (s)')
    ylabel('Amplitude (uV)')
    if i==1
        title('Average stimulus responses (in-ear)')
    end
    if i==2
        title('Average stimulus responses (forehead)')
    end
    if i==3
        title('Average stimulus responses (temporal)')
    end
    if i==4
        title('Average stimulus responses (occipital)')
    end
    plot(timeax(1: 1 + 0.7*fs),normal(i,:))
    plot(timeax(1: 1 + 0.7*fs),deviant(i,:))
    plot(timeax(1: 1 + 0.7*fs),deviant(i,:)-normal(i,:))
    legend('Non-target','Target','MMN')
end

%% ASSR

f2= figure(2);
set(f2,'color','w')

% Plot the average PSD for each channel
for i=1:4
    subplot(1,4,i)
    hold all; grid on;
    if i==1 
        title('PSD (in-ear)')
    end
    if i==2 
        title('PSD (forehead)')
    end
    if i==3 
        title('PSD (temporal)')
    end
    if i==4 
        title('PSD (occipital)')
    end
    ylabel('Amplitude (uV)')
    xlabel('Frequency (Hz)')
    plot(assr_f,assr(i,:));
    xlim([0 100]);
end

%% Alpha Modulation

f3= figure(3);
set(f3,'color','w')

% Plot EC Band Power for each channel
% In-ear
subplot(4,2,1)
hold all; grid on;
title('Frequency Band Power EC (in-ear)')
ylabel('Power (uV)^2/Hz')
xlabel('Frequency Bands')
h1=bar(["Delta","Theta","Alpha","Beta","Gamma"],ec(1,:));
colourbar(h1)

% Forehead
subplot(4,2,3)
hold all; grid on;
title('Frequency Band Power EC (forehead)')
ylabel('Power (uV)^2/Hz')
xlabel('Frequency Bands')
h1=bar(["Delta","Theta","Alpha","Beta","Gamma"],ec(2,:));
colourbar(h1)

% Temporal
subplot(4,2,5)
hold all; grid on;
title('Frequency Band Power EC (temporal)')
ylabel('Power (uV)^2/Hz')
xlabel('Frequency Bands')
h1=bar(["Delta","Theta","Alpha","Beta","Gamma"],ec(3,:));
colourbar(h1)

% Occipital
subplot(4,2,7)
hold all; grid on;
title('Frequency Band Power EC (ocipital)')
ylabel('Power (uV)^2/Hz')
xlabel('Frequency Bands')
h1=bar(["Delta","Theta","Alpha","Beta","Gamma"],ec(4,:));
colourbar(h1)

Plot average EO Band Power for every channel
In-ear
subplot(4,2,2)
hold all; grid on;
title('Frequency Band Power EO (in-ear)')
ylabel('Power (uV)^2/Hz')
xlabel('Frequency Bands')
h1=bar(["Delta","Theta","Alpha","Beta","Gamma"],eo(1,:));
colourbar(h1)

% Forehead
subplot(4,2,4)
hold all; grid on;
title('Frequency Band Power EO (forehead)')
ylabel('Power (uV)^2/Hz')
xlabel('Frequency Bands')
h1=bar(["Delta","Theta","Alpha","Beta","Gamma"],eo(2,:));
colourbar(h1)

% Temporal
subplot(4,2,6)
hold all; grid on;
title('Frequency Band Power EO (temporal)')
ylabel('Power (uV)^2/Hz')
xlabel('Frequency Bands')
h1=bar(["Delta","Theta","Alpha","Beta","Gamma"],eo(3,:));
colourbar(h1)

% Occipital
subplot(4,2,8)
hold all; grid on;
title('Frequency Band Power EO (occipital)')
ylabel('Power (uV)^2/Hz')
xlabel('Frequency Bands')
h1=bar(["Delta","Theta","Alpha","Beta","Gamma"],eo(4,:));
colourbar(h1)

% Plot average spectrogram for each channel

f4 = figure(4);
set(f4, 'color','w');
hold on;

% In-ear
%subplot(1,4,1)
hold on;
[P,F,T]=pspectrum(spect(1,1:75000),250,'spectrogram');
[P2,F2,T2]=pspectrum(spect(1,75001:end),250,'spectrogram');
ylim([0 50])
surf(T/60, F, log10(P), 'EdgeColor', 'none'); 
surf(T(end)/60+T2/60, F2, log10(P2), 'EdgeColor', 'none'); 
title('Alpha Modulation Spectrogram (in-ear)')
xlabel('Time (minutes)')
ylabel('Frequency (Hz)')
c=colorbar
%clim([-20 0]);
ylabel(c, 'Power (dB)')
xlim([0 10])
plot([5 5],[0 50],'Color','k')
xticks(1:10)
text(2,45,'EC','FontSize',14,'FontName','Times New Roman')
text(7,45,'EO','FontSize',14, 'FontName','Times New Roman')



% Forehead
subplot(1,4,2)
hold on;
[P,F,T]=pspectrum(spect(2,:),250,'spectrogram');
ylim([0 50])
surf(T/60, F, log10(P), 'EdgeColor', 'none'); 
title('Alpha Modulation Spectrogram (Forehead)')
xlabel('Time (minutes)')
ylabel('Frequency (Hz)')
c=colorbar
%clim([-20 0]);
ylabel(c, 'Power (dB)')
xlim([0 10])
plot([5 5],[0 50],'Color','k')
xticks(1:10)
text(2,45,'EC','FontSize',14,'FontName','Times New Roman')
text(7,45,'EO','FontSize',14, 'FontName','Times New Roman')

% Temporal
subplot(1,4,3)
hold on;
[P,F,T]=pspectrum(spect(3,:),250,'spectrogram');
ylim([0 50])
surf(T/60, F, log10(P), 'EdgeColor', 'none'); 
title('Alpha Modulation Spectrogram (Temporal)')
xlabel('Time (minutes)')
ylabel('Frequency (Hz)')
c=colorbar
%clim([-20 0]);
ylabel(c, 'Power (dB)')
xlim([0 10])
plot([5 5],[0 50],'Color','k')
xticks(1:10)
text(2,45,'EC','FontSize',14,'FontName','Times New Roman')
text(7,45,'EO','FontSize',14, 'FontName','Times New Roman')

% Occipital
subplot(1,4,4)
hold on;
[P,F,T]=pspectrum(spect(4,:),250,'spectrogram');
ylim([0 50])
surf(T/60, F, log10(P), 'EdgeColor', 'none'); 
title('Alpha Modulation Spectrogram (Occipital)')
xlabel('Time (minutes)')
ylabel('Frequency (Hz)')
c=colorbar
%clim([-20 0]);
ylabel(c, 'Power (dB)')
xlim([0 10])
plot([5 5],[0 50],'Color','k')
xticks(1:10)
text(2,45,'EC','FontSize',14,'FontName','Times New Roman')
text(7,45,'EO','FontSize',14, 'FontName','Times New Roman')

% Quantify in-ear drop in alpha band (t-test)
difference = abs(ec(1,:) - eo(1,:));
db_drop= abs(20*log10(difference))
[h,p]= ttest2(difference([1,3:5]),difference(2)); 

% Quantify in-ear drop at 10Hz specifically (t-test)
[f,pec] = psdplot(spect(1,1:75000),250);
[f,peo]=psdplot(spect(1,75001:end),250);
difference= pec-peo;
freq_10hz_index = find(abs(f - 10) == min(abs(f - 10)));
[h,p,~,stats]=ttest2(difference([1:freq_10hz_index-1,freq_10hz_index+1:end]),difference(freq_10hz_index)); 
20*log10(difference(freq_10hz_index)) % db drop at alpha

%% SSVEP

% Plot average PSD for all channels
f5= figure(5);
set(f5, 'color','w')    

for i=1:4
    subplot(1,4,i)
    hold all; grid on;
    if i==1 
        title('PSD (in-ear)')
    end
    if i==2 
        title('PSD (forehead)')
    end
    if i==3 
        title('PSD (temporal)')
    end
    if i==4 
        title('PSD (occipital)')
    end
    ylabel('Amplitude (uV)')
    xlabel('Frequency (Hz)')
    plot(ssvep_f,ssvep(i,:));
    xlim([0 100]);
end

%% Functions

% Averaging function
function [normal_responses, deviant_responses, assr, assr_f, ec, eo, ec_sd, eo_sd,spect, ssvep, ssvep_f] = average_trials(session_names)
    
    % Montage specifications and sampling frequency
    ear1 = 2;
    ear2 = 3;
    forehead = 4;
    temporal = 5;
    occipital = 6;
    fs=250;
    
    %% P300
    
    % Initialise matrix to store responses for the 3 trials
    all_ears_normal = [];
    all_ears_deviant = [];
    all_forehead_normal = [];
    all_forehead_deviant = [];
    all_temporal_normal = [];
    all_temporal_deviant = [];
    all_occipital_normal = [];
    all_occipital_deviant = [];
    
    % Initialise matrix to store data for the 3 trials
    channel1 = [];
    channel2 = [];
    forehead_filtered = [];
    temporal_filtered = [];
    occipital_filtered = [];
    markers = [];
    
    for i= 1:length(session_names) % For all sessions

        % Set path and load data
        path = 'C:\Users\rmexi\PycharmProjects\BrainFlow\.venv\Scripts';
        path = fullfile(path, ['\',session_names{i}, '\']);
        fullFilePath = fullfile(path, ['data_p300_raw_', session_names{i}, '.csv']);
        load(fullFilePath)

        % Collate the 3 data for all three sessions
        eval(['channel1 = [channel1; filtering(data_p300_raw_' session_names{i} '(1:30500,ear1),fs)''];']);
        eval(['channel2 = [channel2; filtering(data_p300_raw_' session_names{i} '(1:30500,ear2),fs)''];']);
        eval(['forehead_filtered = [forehead_filtered; filtering(data_p300_raw_' session_names{i} '(1:30500,forehead),fs)''];']);
        eval(['temporal_filtered = [temporal_filtered; filtering(data_p300_raw_' session_names{i} '(1:30500,temporal),fs)''];']);
        eval(['occipital_filtered = [occipital_filtered; filtering(data_p300_raw_' session_names{i} '(1:30500,occipital),fs)''];']);
        eval(['markers = [markers; data_p300_raw_' session_names{i} '(1:30500,24)''];']);
       
        eegsize = size(channel1);
        av_channel = (channel1 + channel2)./2;
    
        % Initialise matrices to store IN-EAR stimulus windows            
        epoch = [];
        normal = [];
        deviant = [];
        
        % Loop over every data point exlcuding the first 100ms and last 600ms
        for j = 0.1*fs:eegsize(1,2)-0.6*fs
            % Identify stimulus onset
            if markers(i,j) == 500 || markers(i,j) == 1000
                % Extract windows from 100ms pre stimulus to 600ms post-stimulus
                window = av_channel(i,j-  0.1*fs: j + 0.6*fs); 
                %Baseline Voltage Compensation 
                baseline = mean(av_channel(i,j-0.1*fs:j));
                p3a = window - baseline; % Subtract 0.1s pre-stimulus mean voltage
                epoch = [epoch; p3a];
            end
                
            % Store non-target response       
            if markers(i,j) == 500  
                normal = [normal; p3a];
            end
        
            % Store target resposne
            if markers(i,j) == 1000   
                deviant = [deviant; p3a];
            end
        end
        
        % Save average in-ear responses to matrix
        all_ears_normal = [all_ears_normal; mean(normal)];
        all_ears_deviant = [all_ears_deviant; mean(deviant)];
        
        % Initialise matrices to store FOREHEAD stimulus windows  
        epoch = [];
        normal = [];
        deviant = [];
        
        % Loop over every data point exlcuding the first 100ms and last 600ms
        for j = 0.1*fs:eegsize(1,2)-0.6*fs
            % Identify stimulus onset
            if markers(i,j) == 500 || markers(i,j) == 1000
                % Extract windows from 100ms pre stimulus to 600ms post-stimulus
                window = forehead_filtered(i,j-  0.1*fs: j + 0.6*fs); 
                %Baseline Voltage Compensation 
                baseline = mean(forehead_filtered(i,j-0.1*fs:j));
                p3a = window - baseline; % Subtract 0.1s pre-stimulus mean voltage                epoch = [epoch; p3a];
            
                % Store non target responses
                if markers(i,j) == 500 
                    normal = [normal; p3a];
                end
            
                % Store target responses
                if markers(i,j) == 1000 
                    deviant = [deviant; p3a];
                end
            end 
        end
        
        % Save average deviant and non deviant forehead responses
        all_forehead_normal = [all_forehead_normal; mean(normal)];
        all_forehead_deviant = [all_forehead_deviant; mean(deviant)];
        
        
        % Initialise matrices to store TEMPORAL stimulus windows 
        epoch = [];
        normal = [];
        deviant = [];
        
        % Loop over every data point exlcuding the first 100ms and last 600ms
        for j = 0.1*fs:eegsize(1,2)-0.6*fs
            % Identify stimulus onset
            if markers(i,j) == 500 || markers(i,j) == 1000
                % Extract windows from 100ms pre stimulus to 600ms post-stimulus
                window = temporal_filtered(i,j-  0.1*fs: j + 0.6*fs); 
                % Baseline Voltage Compensation 
                baseline = mean(temporal_filtered(i,j-0.1*fs:j));
                p3a = window - baseline; % Subtract 0.1s pre-stimulus mean voltage
                epoch = [epoch; p3a];
            
                % Save non target response
                if markers(i,j) == 500   
                    normal = [normal; p3a];
                end
            
                % Save target response
                if markers(i,j) == 1000   
                   deviant = [deviant; p3a];
                end
            end  
        end
        
        % Save average deviant and non deviant response
        all_temporal_normal = [all_temporal_normal; mean(normal)];
        all_temporal_deviant = [all_temporal_deviant; mean(deviant)];

        % Initialise matrices to store OCCIPITAL stimulus windows 
        epoch = [];
        normal = [];
        deviant = [];
        
        % Loop over every data point exlcuding the first 100ms and last 600ms
        for j = 0.1*fs:eegsize(1,2)-0.6*fs
            % Identify stimulus onset
            if markers(i,j) == 500 || markers(i,j) == 1000
                % Extract windows from 100ms pre stimulus to 600ms post-stimulus
                window = occipital_filtered(i,j-  0.1*fs: j + 0.6*fs); 
                % Baseline Voltage Compensation 
                baseline = mean(occipital_filtered(i,j-0.1*fs:j));
                p3a = window - baseline; % Subtract 0.1s pre-stimulus mean voltage
                epoch = [epoch; p3a];
            
                % Store non target resposne
                if markers(i,j) == 500 
                   normal = [normal; p3a];
                end
            
                % Store target response
                if markers(i,j) == 1000 
                    deviant = [deviant; p3a];
                end
            end
        end
        
       % Store average deviant and non deviant responses
       all_occipital_normal = [all_occipital_normal; mean(normal)];
       all_occipital_deviant = [all_occipital_deviant; mean(deviant)];
        
    end
    
    % Compute mean of all 3 trials
    all_ears_normal = mean(all_ears_normal);
    all_ears_deviant = mean(all_ears_deviant);
    all_forehead_normal = mean(all_forehead_normal);
    all_forehead_deviant = mean(all_forehead_deviant);
    all_temporal_normal = mean(all_temporal_normal);
    all_temporal_deviant = mean(all_temporal_deviant);
    all_occipital_normal = mean(all_occipital_normal);
    all_occipital_deviant = mean(all_occipital_deviant);

    % Collate average responses into one matrix
    normal_responses = [all_ears_normal; all_forehead_normal; all_temporal_normal; all_occipital_normal];
    deviant_responses = [all_ears_deviant; all_forehead_deviant; all_temporal_deviant; all_occipital_deviant];
    
    
    %% ASSR
    
    % Initialise matrix to store all 3 trials data
    all_channel1 = [];
    all_channel2 = [];
    all_forehead = [];
    all_temporal = [];
    all_occipital = [];
    
    for i= 1:length(session_names) % For every trial
        % Set path and load data
        path = 'C:\Users\rmexi\PycharmProjects\BrainFlow\.venv\Scripts';
        path = fullfile(path, ['\',session_names{i}, '\']);
        fullFilePath = fullfile(path, ['data_assr_raw_', session_names{i}, '.csv']);
        load(fullFilePath)
        % Collate the data for every session
        eval(['all_channel1 = [all_channel1; filtering(data_assr_raw_' session_names{i} '(1:23157,ear1),fs)''];']);
        eval(['all_channel2 = [all_channel2; filtering(data_assr_raw_' session_names{i} '(1:23157,ear2),fs)''];']);
        eval(['all_forehead = [all_forehead; filtering(data_assr_raw_' session_names{i} '(1:23157,forehead),fs)''];']);
        eval(['all_temporal = [all_temporal; filtering(data_assr_raw_' session_names{i} '(1:23157,temporal),fs)''];']);
        eval(['all_occipital = [all_occipital; filtering(data_assr_raw_' session_names{i} '(1:23157,occipital),fs)''];']);
    end
    
    all_ear = (all_channel1 + all_channel2)./2;

    % Compute In-ear PSD for the 3 trials and average
    [f,psd1]= psdplot(all_ear(1,500:end),fs);
    [~,psd2]= psdplot(all_ear(2,500:end),fs);
    [~,psd3]= psdplot(all_ear(3,500:end),fs);
    ear_psd = [psd1;psd2;psd3];
    assr_ear = mean(ear_psd);

    % Compute Forehead PSD for the 3 trials and average
    [~,psd1]= psdplot(all_forehead(1,500:end),fs);
    [~,psd2]= psdplot(all_forehead(2,500:end),fs);
    [~,psd3]= psdplot(all_forehead(3,500:end),fs);
    forehead_psd = [psd1;psd2;psd3];
    assr_forehead = mean(forehead_psd);

    % Compute Temporal PSD for the 3 trials and average
    [~,psd1]= psdplot(all_temporal(1,500:end),fs);
    [~,psd2]= psdplot(all_temporal(2,500:end),fs);
    [~,psd3]= psdplot(all_temporal(3,500:end),fs);
    temporal_psd = [psd1;psd2;psd3];
    assr_temporal = mean(temporal_psd);

    % Compute Occipital PSD for the 3 trials and average
    [~,psd1]= psdplot(all_occipital(1,500:end),fs);
    [~,psd2]= psdplot(all_occipital(2,500:end),fs);
    [~,psd3]= psdplot(all_occipital(3,500:end),fs);
    occipital_psd = [psd1;psd2;psd3];
    assr_occipital = mean(occipital_psd);

    % Store all average PSDs in one matrix
    assr = [assr_ear; assr_forehead; assr_temporal; assr_occipital];
    assr_f = f;
    
    %% Alpha Modulation

    % EC
    % Initialise matrix to store all trials data
    all_channel1 = [];
    all_channel2 = [];
    all_forehead = [];
    all_temporal = [];
    all_occipital = [];
    
    for i= 1:length(session_names) % For every trial
        % Set path and load data
        path = 'C:\Users\rmexi\PycharmProjects\BrainFlow\.venv\Scripts';
        path = fullfile(path, ['\',session_names{i}, '\']);
        fullFilePath = fullfile(path, ['data_ec_raw_', session_names{i}, '.csv']);
        load(fullFilePath)
        % Collate all 3 trials for each channel
        eval(['all_channel1 = [all_channel1; filtering(data_ec_raw_' session_names{i} '(1:75000,ear1),fs)''];']);
        eval(['all_channel2 = [all_channel2; filtering(data_ec_raw_' session_names{i} '(1:75000,ear2),fs)''];']);
        eval(['all_forehead = [all_forehead; filtering(data_ec_raw_' session_names{i} '(1:75000,forehead),fs)''];']);
        eval(['all_temporal = [all_temporal; filtering(data_ec_raw_' session_names{i} '(1:75000,temporal),fs)''];']);
        eval(['all_occipital = [all_occipital; filtering(data_ec_raw_' session_names{i} '(1:75000,occipital),fs)''];']);
    end
    
    all_ears = (all_channel1 + all_channel2)./2;
    
    % Compute In-ear band power for 3 trials and average
    bands1 = bandextract(all_ears(1,500:end),fs);
    bands2 = bandextract(all_ears(2,500:end),fs);
    bands3 = bandextract(all_ears(3,500:end),fs);
    bands = [bands1;bands2;bands3];
    sd_ears = std(bands);
    bands_ear = mean(bands);
    
    % Compute Forehead band power for 3 trials and average
    bands1 = bandextract(all_forehead(1,500:end),fs);
    bands2 = bandextract(all_forehead(2,500:end),fs);
    bands3 = bandextract(all_forehead(3,500:end),fs);
    bands = [bands1;bands2;bands3];
    sd_forehead = std(bands);
    bands_forehead = mean(bands);

    % Compute Temporal band power for 3 trials and average
    bands1 = bandextract(all_temporal(1,500:end),fs);
    bands2 = bandextract(all_temporal(2,500:end),fs);
    bands3 = bandextract(all_temporal(3,500:end),fs);
    bands = [bands1;bands2;bands3];
    sd_temporal = std(bands);
    bands_temporal = mean(bands);
    
    % Compute Occipital band power for 3 trials and average
    bands1 = bandextract(all_occipital(1,500:end),fs);
    bands2 = bandextract(all_occipital(2,500:end),fs);
    bands3 = bandextract(all_occipital(3,500:end),fs);
    bands = [bands1;bands2;bands3];
    sd_occipital = std(bands);
    bands_occipital = mean(bands);
    
    % Store average band power in one matrix
    ec = [bands_ear; bands_forehead; bands_temporal; bands_occipital];
    ec_sd = [sd_ears;sd_forehead;sd_temporal;sd_occipital];

    % Store average frequency domain data into one matrix
    spect_ec= [mean(all_channel1); mean(all_channel2); mean(all_forehead); mean(all_temporal); mean(all_occipital)];
    
    % EO
    % Initialise matrix to store all trials data
    all_channel1 = [];
    all_channel2 = [];
    all_forehead = [];
    all_temporal = [];
    all_occipital = [];
    
    for i= 1:length(session_names) % For every trial
        % Set path and load data
        path = 'C:\Users\rmexi\PycharmProjects\BrainFlow\.venv\Scripts';
        path = fullfile(path, ['\',session_names{i}, '\']);
        fullFilePath = fullfile(path, ['data_eo_raw_', session_names{i}, '.csv']);
        load(fullFilePath)
        % Collate every trial for each channel
        eval(['all_channel1 = [all_channel1; filtering(data_eo_raw_' session_names{i} '(1:75000,ear1),fs)''];']);
        eval(['all_channel2 = [all_channel2; filtering(data_eo_raw_' session_names{i} '(1:75000,ear2),fs)''];']);
        eval(['all_forehead = [all_forehead; filtering(data_eo_raw_' session_names{i} '(1:75000,forehead),fs)''];']);
        eval(['all_temporal = [all_temporal; filtering(data_eo_raw_' session_names{i} '(1:75000,temporal),fs)''];']);
        eval(['all_occipital = [all_occipital; filtering(data_eo_raw_' session_names{i} '(1:75000,occipital),fs)''];']);
    end
    
    all_ears = (all_channel1 + all_channel2)./2;
    
    % Compute In-ear band power and average
    bands1 = bandextract(all_ears(1,500:end),fs);
    bands2 = bandextract(all_ears(2,500:end),fs);
    bands3 = bandextract(all_ears(3,500:end),fs);
    bands = [bands1;bands2;bands3];
    sd_ears = std(bands);
    bands_ear = mean(bands);
    
    % Compute Forehead band power and average
    bands1 = bandextract(all_forehead(1,500:end),fs);
    bands2 = bandextract(all_forehead(2,500:end),fs);
    bands3 = bandextract(all_forehead(3,500:end),fs);
    bands = [bands1;bands2;bands3];
    sd_forehead = std(bands);
    bands_forehead = mean(bands);
    
    % Compute Temporal band power and average
    bands1 = bandextract(all_temporal(1,500:end),fs);
    bands2 = bandextract(all_temporal(2,500:end),fs);
    bands3 = bandextract(all_temporal(3,500:end),fs);
    bands = [bands1;bands2;bands3];
    sd_temporal = std(bands);
    bands_temporal = mean(bands);
    
    % Compute Occipital band power and average
    bands1 = bandextract(all_occipital(1,500:end),fs);
    bands2 = bandextract(all_occipital(2,500:end),fs);
    bands3 = bandextract(all_occipital(3,500:end),fs);
    bands = [bands1;bands2;bands3];
    sd_occipital = std(bands);
    bands_occipital = mean(bands);
    
    % Store band power averages in one matrix
    eo = [bands_ear; bands_forehead; bands_temporal; bands_occipital];
    eo_sd = [sd_ears;sd_forehead;sd_temporal;sd_occipital];


    % Store average frequency domain data in one matrix
    spect_eo= [mean(all_channel1); mean(all_channel2); mean(all_forehead); mean(all_temporal); mean(all_occipital)];
    spect = [spect_ec, spect_eo];

    %% SSVEP
    
    % Initialise matrix to store data from all trials
    all_channel1 = [];
    all_channel2 = [];
    all_forehead = [];
    all_temporal = [];
    all_occipital = [];
    
    
    for i= 1:length(session_names) % For every trial
        % Set path and load data
        path = 'C:\Users\rmexi\PycharmProjects\BrainFlow\.venv\Scripts';
        path = fullfile(path, ['\',session_names{i}, '\']);
        fullFilePath = fullfile(path, ['data_ssvep_raw_', session_names{i}, '.csv']);
        load(fullFilePath)
        % Collate every trial
        eval(['all_channel1 = [all_channel1; trials(filtering(data_ssvep_raw_' session_names{i} '(1:15470,ear1),fs)'')];']);
        eval(['all_channel2 = [all_channel2; trials(filtering(data_ssvep_raw_' session_names{i} '(1:15470,ear2),fs)'')];']);
        eval(['all_forehead = [all_forehead; trials(filtering(data_ssvep_raw_' session_names{i} '(1:15470,forehead),fs)'')];']);
        eval(['all_temporal = [all_temporal; trials(filtering(data_ssvep_raw_' session_names{i} '(1:15470,temporal),fs)'')];']);
        eval(['all_occipital = [all_occipital; trials(filtering(data_ssvep_raw_' session_names{i} '(1:15470,occipital),fs)'')];']);
    end
    
    all_ear = (all_channel1 + all_channel2)./2;
    
    % Compute In-ear PSD for each trial and average
    [f,psd1]= psdplot(all_ear(1,:),fs);
    [~,psd2]= psdplot(all_ear(2,:),fs);
    [~,psd3]= psdplot(all_ear(3,:),fs);
    ear_psd = [psd1;psd2;psd3];
    ssvep_ear = mean(ear_psd);
    
    % Compute Forehead PSD for each trial and average
    [~,psd1]= psdplot(all_forehead(1,:),fs);
    [~,psd2]= psdplot(all_forehead(2,:),fs);
    [~,psd3]= psdplot(all_forehead(3,:),fs);
    forehead_psd = [psd1;psd2;psd3];
    ssvep_forehead = mean(forehead_psd);
    
    % Compute Temporal PSD for each trial and average
    [~,psd1]= psdplot(all_temporal(1,:),fs);
    [~,psd2]= psdplot(all_temporal(2,:),fs);
    [~,psd3]= psdplot(all_temporal(3,:),fs);
    temporal_psd = [psd1;psd2;psd3];
    ssvep_temporal = mean(temporal_psd);
    
    % Compute Occipital PSD for each trial and averag
    [~,psd1]= psdplot(all_occipital(1,:),fs);
    [~,psd2]= psdplot(all_occipital(2,:),fs);
    [~,psd3]= psdplot(all_occipital(3,:),fs);
    occipital_psd = [psd1;psd2;psd3];
    ssvep_occipital = mean(occipital_psd);
    
    % Store all averages in one matrix
    ssvep = [ssvep_ear; ssvep_forehead; ssvep_temporal; ssvep_occipital];
    ssvep_f = f;
end

function filtered_data = filtering(data,fs)
    %DC offset at 0Hz, Noise and muscle artifacts above 50Hz
    %4th order Bandpass between 1 and 100Hz 
    [b,a] = butter(4,[2 100]*2/fs,'bandpass');

    %Power Supply Interferance at 50Hz
    %4th order Notch filter between 45 and 55Hz
    [b2,a2] = butter(4,[45 55]*2/fs,'stop');

    %Filtfilt removes transient response of filters
    %filtered_data = filtfilt(b2, a2, filtfilt(b,a,data));
    filtered_data = filter(b2, a2, filter(b,a,data));

    % Moving average filter
    MA_coef_num = 5; 
    MA = ones(1,MA_coef_num)/MA_coef_num;
    filtered_data = conv(filtered_data, MA, 'same');

end

function [x_ax,y_ax] = psdplot(channel,fs)
    %Extract Periodogram in uV 
    psd= fft(channel);
    x_ax= 0:fs/length(psd):fs-fs/length(psd);
    y_ax= abs(psd)./length(psd); %abs psd normalized by signal length
end

function bands = bandextract(channel,fs)
    %Extract band power in (uV)^2/Hz
    delta = bandpower(channel,fs,[1 4]);
    theta = bandpower(channel,fs,[4 9]);
    alpha = bandpower(channel,fs,[9 14]);
    beta = bandpower(channel,fs,[14 33]);
    gamma = bandpower(channel,fs,[33 100]);  
    bands= [delta, theta, alpha, beta, gamma];
end

function colourbar(h1)
    %Change the colours of the bars
    h1.FaceColor = 'flat';
    h1.CData(1,:) = [1 0 0];
    h1.CData(2,:) = [1 0.9 0.1];
    h1.CData(3,:) = [0 1 0];
    h1.CData(4,:) = [0 0 1];
    h1.CData(5,:) = [0.5 0.2 1];
end

function average_trial = trials(data)
    % Separate the 5 SSVEP trials
    trials=[];
    for i= 1:length(data)/5 - 1:length(data)-length(data)/5
        trials = [trials; data(i:i+length(data)/5)];
    end
    average_trial = mean(trials);
end