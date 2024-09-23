clc
clear

% COMPUTING GRAND AVERAGES AND STATISTICAL TESTING

%% Compute in-subject averages

% Rocio
session_names = {'18_02_24_rocio1', '18_02_24_rocio2', '18_02_24_rocio3'};
[normal_rocio, deviant_rocio, assr_rocio, assr_f_rocio, ec_rocio, eo_rocio, spect_rocio, ssvep_rocio, ssvep_f_rocio] = average_trials(session_names);

% Ceci
session_names = {'19_02_24_ceci1', '19_02_24_ceci2', '19_02_24_ceci3'};
[normal_ceci, deviant_ceci, assr_ceci, assr_f_ceci, ec_ceci, eo_ceci, spect_ceci, ssvep_ceci, ssvep_f_ceci] = average_trials(session_names);

% Aaron
session_names = {'19_02_24_aaron1', '19_02_24_aaron2', '19_02_24_aaron3'};
[normal_aaron, deviant_aaron, assr_aaron, assr_f_aaron, ec_aaron, eo_aaron, spect_aaron, ssvep_aaron, ssvep_f_aaron] = average_trials(session_names);

% August
session_names = {'22_02_24_august1', '22_02_24_august2', '22_02_24_august3'};
[normal_august, deviant_august, assr_august, assr_f_august, ec_august, eo_august, spect_august, ssvep_august, ssvep_f_august] = average_trials(session_names);

% Leire
session_names = {'22_02_24_leire1', '22_02_24_leire2', '22_02_24_leire3'};
[normal_leire, deviant_leire, assr_leire, assr_f_leire, ec_leire, eo_leire, spect_leire, ssvep_leire, ssvep_f_leire] = average_trials(session_names);

% Josh
session_names = {'23_02_24_josh1', '23_02_24_josh2', '23_02_24_josh3'};
[normal_josh, deviant_josh, assr_josh, assr_f_josh, ec_josh, eo_josh, spect_josh, ssvep_josh, ssvep_f_josh] = average_trials(session_names);

%% P300

% Collate all normal and all deviant subject averaged responses for in-ear
normal = [normal_rocio(1,:); normal_ceci(1,:); normal_aaron(1,:); normal_august(1,:); normal_leire(1,:); normal_josh(1,:)];
deviant = [deviant_rocio(1,:); deviant_ceci(1,:); deviant_aaron(1,:); deviant_august(1,:); deviant_leire(1,:); deviant_josh(1,:)];

% Plotting grand-averaged normal and deviant responses
mean_normal = mean(normal);
mean_deviant = mean(deviant);
MA_coef_num = 12; 
MA = ones(1,MA_coef_num)/MA_coef_num;
mean_normal = conv(mean_normal, MA, 'same');
mean_deviant = conv(mean_deviant, MA, 'same');
fs= 250;
time_ax = 0: 1/fs : (length(mean_normal)-1)/fs;
f5= figure(5);
set(f5, 'color','w')
hold all;
plot(time_ax,mean_normal)
plot(time_ax,mean_deviant)
len= length(time_ax(0.18*250:0.35*250));
fill([time_ax(0.18*250:0.35*250) fliplr(time_ax(0.18*250:0.35*250))],[3*ones(1,len) -1*ones(1,len)],'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none')
xlabel('Time (s)')
ylabel('Voltage (\muV)')
legend('Non-target','Target')

% T-test
difference = mean_normal-mean_deviant;
[h_av, p_av, ci_av, stats_av] = ttest(difference');

if h_av == 1
    fprintf('There is a significant difference between target and non-target responses (p = %.3f).\n', p_av);
else
    fprintf('There is no significant difference between target and non-target responses (p = %.3f).\n', p_av);
end

%% ASSR

% Collate all in-subject average PSDs for in-ear
all_assr = [assr_rocio(1,:); assr_ceci(1,:); assr_aaron(1,:); assr_august(1,:); assr_leire(1,:); assr_josh(1,:)];

% Compute and plot grand-average assr response
mean_assr = mean(all_assr);

f2=figure(2);
set(f2, 'color','w')
hold all;
plot(assr_f_rocio, mean_assr)
xlim([0 100])
%title('ASSR Periodogram')
xlabel('Frequency (Hz)')
ylabel('PSD (\muV^2/Hz)')

% Find peaks between 35Hz and 45Hz
target_value = 35;
[~, index0] = min(abs(assr_f_rocio - target_value));
target_value = 45;
[~, index1] = min(abs(assr_f_rocio - target_value));
[peak_magnitude, index_peak, width] = findpeaks(mean_assr(index0:index1));

% Identify largest peak
[max_magnitude, max_index]=max(peak_magnitude);
peak_location0 = index_peak(max_index);
peak_width = width(max_index);
scatter(assr_f_rocio(peak_location0+index0),max_magnitude, 20, 'filled','red')

% Find second harmonic peaks
target_value = 2*assr_f_rocio(peak_location0+index0)-3;
[~, index2] = min(abs(assr_f_rocio - target_value));
target_value = 2*assr_f_rocio(peak_location0+index0) +3;
[~, index3] = min(abs(assr_f_rocio - target_value));
[peak_magnitude, index_peak] = findpeaks(mean_assr(index2:index3));

% Identify largest peak
[max_magnitude, max_index]=max(peak_magnitude);
peak_location = index_peak(max_index);
scatter(assr_f_rocio(peak_location+index2),max_magnitude, 20, 'filled','green')


[~, index40] = min(abs(assr_f_rocio - 40));
[~, index35] = min(abs(assr_f_rocio - 35));
[~, index45] = min(abs(assr_f_rocio - 45));

signal= mean_assr(index40);
assr_no40= [mean_assr(index35:index40-1) mean_assr(index40+1:index45)];
noise = mean(assr_no40);
20*log10(signal/noise)

%% Alpha Modulation

% Collate all in-subject averages band power for in-ear
all_ec = [ec_rocio(1,:); ec_ceci(1,:); ec_august(1,:); ec_leire(1,:); ec_josh(1,:)];
all_eo = [eo_rocio(1,:); eo_ceci(1,:); eo_august(1,:); eo_leire(1,:); eo_josh(1,:)];

% Compute grand-average band powers
mean_ec = mean(all_ec);
mean_eo = mean(all_eo);

% Compute standard deviation in grand-average
std_ec = std(all_ec);
std_eo = std(all_eo);
lower_error = zeros(1,5);
bands = ["Delta","Theta","Alpha","Beta","Gamma"];

% Plot grand average band power with error bars
f6 = figure(6);
set(f6, 'color','w')

%EC
subplot(1,2,1)
hold all;
h1=bar([1,2,3,4,5],mean_ec);
errorbar([1,2,3,4,5],mean_ec, lower_error, std_ec, 'k.', 'LineWidth', 0.6, 'CapSize', 10);
set(gca, 'xtick', [1,2,3,4,5], 'xticklabel', bands)
title('Frequency Band Power EC')
ylabel('Power (uV)^2/Hz')
xlabel('Frequency Bands')
colourbar(h1)

%EO
subplot(1,2,2)
hold all;
h1=bar([1,2,3,4,5],mean_eo);
errorbar([1,2,3,4,5], mean_eo, lower_error, std_eo, 'k.', 'LineWidth', 0.6, 'CapSize', 10);
set(gca, 'xtick', [1,2,3,4,5], 'xticklabel', bands)
title('Frequency Band Power EO')
ylabel('Power (uV)^2/Hz')
xlabel('Frequency Bands')
colourbar(h1)

% T-test between ec and eo for all bands
[h, p, ci, stats] = ttest(all_ec, all_eo);

if h == 1
    fprintf('There is a significant difference between eyes closed and eyes open (p = %.3f).\n', p);
else
    fprintf('There is no significant difference between eyes closed and eyes open responses (p = %.3f).\n', p);
end

% Collate all in-subject average spectrograms for in-ear
all_spect_ec = [spect_rocio(1,1:75000); spect_ceci(1,1:75000); spect_august(1,1:75000); spect_leire(1,1:75000); spect_josh(1,1:75000)];
all_spect_eo = [spect_rocio(1,75001:end); spect_ceci(1,75001:end); spect_august(1,75001:end); spect_leire(1,75001:end); spect_josh(1,75001:end)];

% Compute grand-average spectrogram
mean_spect_ec = mean(all_spect_ec);
mean_spect_eo = mean(all_spect_eo);

% Display grand-average spectrogram
f3= figure(3);
set(f3, 'color', 'w')
subplot(1,2,1)
hold on;
[P,F,T]=pspectrum(mean_spect_ec,250,'spectrogram');
T= T/60;
ylim([0 50])
[P2,F2,T2]=pspectrum(mean_spect_eo,250,'spectrogram');
T2=T2/60;
ylim([0 50])
surf(T, F, log10(P), 'EdgeColor', 'none'); 
surf(T2+T(end)-1/60, F2, log10(P), 'EdgeColor', 'none'); 
title('In-ear electrodes')
xlabel('Time (minutes)')
ylabel('Frequency (Hz)')
c=colorbar
clim([-20 0]);
ylabel(c, 'Power (dB)')
xlim([0 10])
plot([5 5],[0 50],'Color','k')
xticks(1:9)
text(2,45,'EC','FontSize',14,'FontName','Times New Roman')
text(7,45,'EO','FontSize',14, 'FontName','Times New Roman')

% Collate all in-subject average spectrograms for occipital
all_spect_ec = [spect_rocio(4,1:75000); spect_ceci(4,1:75000); spect_august(4,1:75000); spect_leire(4,1:75000); spect_josh(4,1:75000)];
all_spect_eo = [spect_rocio(4,75001:end); spect_ceci(4,75001:end); spect_august(4,75001:end); spect_leire(4,75001:end); spect_josh(4,75001:end)];

% Compute grand-average spectrogram
mean_spect_ec = mean(all_spect_ec);
mean_spect_eo = mean(all_spect_eo);

% Display spectrogram
subplot(1,2,2)
hold on;
[P,F,T]=pspectrum(mean_spect_ec,250,'spectrogram');
T= T/60;
ylim([0 50])
[P2,F2,T2]=pspectrum(mean_spect_eo,250,'spectrogram');
T2=T2/60;
ylim([0 50])
surf(T, F, log10(P), 'EdgeColor', 'none'); 
surf(T2+T(end)-1/60, F2, log10(P), 'EdgeColor', 'none'); 
title('Occipital wet electrode')
xlabel('Time (minutes)')
ylabel('Frequency (Hz)')
c=colorbar
%clim([-70 20]);
xlim([0 10])
ylabel(c, 'Power (dB)')
plot([5 5],[0 50],'Color','k')
xticks(1:9)
text(2,45,'EC','FontSize',14,'FontName','Times New Roman')
text(7,45,'EO','FontSize',14, 'FontName','Times New Roman')
clim([-20 0]);

%% SSVEP

% Collate all in-subject average PSDs for in-ear
all_ssvep = [ssvep_rocio(1,:); ssvep_ceci(1,:); ssvep_aaron(1,:); ssvep_august(1,:); ssvep_leire(1,:); ssvep_josh(1,:)];

% Compute grand-average ssvep PSD
mean_ssvep = mean(all_ssvep);

% Plot grand-average
f4=figure(4);
set(f4, 'color','w')
hold all;
plot(ssvep_f_rocio, mean_ssvep)
xlim([0 100])
xlabel('Frequency (Hz)')
ylabel('A (\muV^2/Hz)')
%title('SSVEP Periodogram')

% Find 10Hz peak
target_value = 7;
[~, index0] = min(abs(ssvep_f_rocio - target_value));
target_value = 13;
[~, index1] = min(abs(ssvep_f_rocio - target_value));
[peak_magnitude, index_peak, width] = findpeaks(mean_ssvep(index0:index1));

% Identify largest peak
[max_magnitude, max_index]=max(peak_magnitude);
peak_location0 = index_peak(max_index);
peak_width = width(max_index);
scatter(ssvep_f_rocio(peak_location0+index0),max_magnitude, 20, 'filled','red')

% SNR
target_value = 5;
[~, index5] = min(abs(ssvep_f_rocio - target_value));
target_value = 15;
[~, index15] = min(abs(ssvep_f_rocio - target_value));
[peak_magnitude, index_peak, width] = findpeaks(mean_ssvep(index5:index15));
peak_index = peak_location0+index0;
power10 = mean_ssvep(peak_index);
power_noise = [mean_ssvep(index5:peak_index-1) mean_ssvep(peak_index+1:index15)];
20*log10(power10/mean(power_noise))

% Find second harmonic peaks 
target_value = 2*ssvep_f_rocio(peak_location0+index0)-3;
[~, index2] = min(abs(ssvep_f_rocio - target_value));
target_value = 2*ssvep_f_rocio(peak_location0+index0) +3;
[~, index3] = min(abs(ssvep_f_rocio - target_value));
[peak_magnitude, index_peak] = findpeaks(mean_ssvep(index2:index3));

% Identify largest peak
[max_magnitude, max_index]=max(peak_magnitude);
peak_location = index_peak(max_index);
scatter(ssvep_f_rocio(peak_location+index2),max_magnitude, 20, 'filled','yellow')

% Find third harmonic peaks 
target_value = 3*ssvep_f_rocio(peak_location0+index0)-3;
[~, index4] = min(abs(ssvep_f_rocio - target_value));
target_value = 3*ssvep_f_rocio(peak_location0+index0) +3;
[~, index5] = min(abs(ssvep_f_rocio - target_value));
[peak_magnitude, index_peak] = findpeaks(mean_ssvep(index4:index5));

% Identify largest peak
[max_magnitude, max_index]=max(peak_magnitude);
peak_location = index_peak(max_index);
scatter(ssvep_f_rocio(peak_location+index4),max_magnitude, 20, 'filled','green')

% Find fourth harmonic peaks 
target_value = 4*ssvep_f_rocio(peak_location0+index0)-3;
[~, index6] = min(abs(ssvep_f_rocio - target_value));
target_value = 4*ssvep_f_rocio(peak_location0+index0) +3;
[~, index7] = min(abs(ssvep_f_rocio - target_value));
[peak_magnitude, index_peak] = findpeaks(mean_ssvep(index6:index7));

% Identify largest peak
[max_magnitude, max_index]=max(peak_magnitude);
peak_location = index_peak(max_index);
scatter(ssvep_f_rocio(peak_location+index6),max_magnitude, 20, 'filled','blue')

% Find fifth harmonic peaks 
target_value = 5*ssvep_f_rocio(peak_location0+index0)-3;
[~, index8] = min(abs(ssvep_f_rocio - target_value));
target_value = 5*ssvep_f_rocio(peak_location0+index0) +3;
[~, index9] = min(abs(ssvep_f_rocio - target_value));
[peak_magnitude, index_peak] = findpeaks(mean_ssvep(index8:index9));

% Identify largest peak
[max_magnitude, max_index]=max(peak_magnitude);
peak_location = index_peak(max_index);
scatter(ssvep_f_rocio(peak_location+index8),max_magnitude, 20, 'filled','cyan')

%% Functions

% Averaging function
function [normal_responses, deviant_responses, assr, assr_f, ec, eo, spect, ssvep, ssvep_f] = average_trials(session_names)
    
    % Montage specificatins and sampling frequency
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
                p3a = window - baseline;
                epoch = [epoch; p3a];
            end
                
            % Store non-target response       
            if markers(i,j) == 500 
                normal = [normal; p3a];
            end
        
            % Store non-target response
            if markers(i,j) == 1000   
                deviant = [deviant; p3a];
            end
        end   
        
        % Save average in-ear responses
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
                p3a = window - baseline;
                epoch = [epoch; p3a];
            
                % Store non target responses
                if markers(i,j) == 500 %Non target stimulus response 
                    normal = [normal; p3a];
                end
            
                % Store target responses
                if markers(i,j) == 1000 %Target stimulus response  
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
                %Breaks down channel data into individual stimuli epochs
                window = temporal_filtered(i,j-  0.1*fs: j + 0.6*fs); 
                % Extract windows from 100ms pre stimulus to 600ms post-stimulus
                baseline = mean(temporal_filtered(i,j-0.1*fs:j));
                p3a = window - baseline;
                epoch = [epoch; p3a];
            
                % Save non target response
                if markers(i,j) == 500 %Non target stimulus response  
                    normal = [normal; p3a];
                end
            
                % Save target response
                if markers(i,j) == 1000 %Target stimulus response  
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
                %Baseline Voltage Compensation (0.1s pre-stimulus mean voltage)
                baseline = mean(occipital_filtered(i,j-0.1*fs:j));
                p3a = window - baseline;
                epoch = [epoch; p3a];
            
                % Store non target resposne
                if markers(i,j) == 500 %Non target stimulus response
                   normal = [normal; p3a];
                end
            
                % Store target resposne
                if markers(i,j) == 1000 %Target stimulus response
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
    bands_ear = mean(bands);
    
    % Compute Forehead band power for 3 trials and average
    bands1 = bandextract(all_forehead(1,500:end),fs);
    bands2 = bandextract(all_forehead(2,500:end),fs);
    bands3 = bandextract(all_forehead(3,500:end),fs);
    bands = [bands1;bands2;bands3];
    bands_forehead = mean(bands);
    
    % Compute Temporal band power for 3 trials and average
    bands1 = bandextract(all_temporal(1,500:end),fs);
    bands2 = bandextract(all_temporal(2,500:end),fs);
    bands3 = bandextract(all_temporal(3,500:end),fs);
    bands = [bands1;bands2;bands3];
    bands_temporal = mean(bands);
    
    % Compute Occipital band power for 3 trials and average
    bands1 = bandextract(all_occipital(1,500:end),fs);
    bands2 = bandextract(all_occipital(2,500:end),fs);
    bands3 = bandextract(all_occipital(3,500:end),fs);
    bands = [bands1;bands2;bands3];
    bands_occipital = mean(bands);
    
    % Store average band power in one matrix
    ec = [bands_ear; bands_forehead; bands_temporal; bands_occipital];

    % Store average frequency domain data into one matrix
    spect_ec= [mean(all_ears); mean(all_forehead); mean(all_temporal); mean(all_occipital)];
    
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
    bands_ear = mean(bands);
    
    % Compute Forehead band power and average
    bands1 = bandextract(all_forehead(1,500:end),fs);
    bands2 = bandextract(all_forehead(2,500:end),fs);
    bands3 = bandextract(all_forehead(3,500:end),fs);
    bands = [bands1;bands2;bands3];
    bands_forehead = mean(bands);
    
    % Compute Temporal band power and average
    bands1 = bandextract(all_temporal(1,500:end),fs);
    bands2 = bandextract(all_temporal(2,500:end),fs);
    bands3 = bandextract(all_temporal(3,500:end),fs);
    bands = [bands1;bands2;bands3];
    bands_temporal = mean(bands);
    
    % Compute Occipital band power and average
    bands1 = bandextract(all_occipital(1,500:end),fs);
    bands2 = bandextract(all_occipital(2,500:end),fs);
    bands3 = bandextract(all_occipital(3,500:end),fs);
    bands = [bands1;bands2;bands3];
    bands_occipital = mean(bands);
    
    % Store band power averages in one matrix
    eo = [bands_ear; bands_forehead; bands_temporal; bands_occipital];

    % Store average frequency domain data in one matrix
    spect_eo= [mean(all_ears); mean(all_forehead); mean(all_temporal); mean(all_occipital)];
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
    
    % Compute Occipital PSD for each trial and average
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
    % psd= fft(channel);
    % x_ax= 0:fs/length(psd):fs-fs/length(psd);
    % y_ax= abs(psd)./length(psd); %abs psd normalized by signal length

    %Extract Periodogram in uV^2 against Hz
    psd= fft(channel);
    x_ax= 0:fs/length(psd):fs-fs/length(psd);
    y_ax= (abs(psd).^2)./fs*length(psd); %abs psd normalized by signal length
    y_ax(2:end-1) = 2*y_ax(2:end-1); % double psd except DC part
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
    % Separate the 5 repetitions of SSVEP
    trials=[];
    for i= 1:length(data)/5 - 1:length(data)-length(data)/5
        trials = [trials; data(i:i+length(data)/5)];
    end
    average_trial = mean(trials);
end