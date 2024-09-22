clc
clear
close

%% Parameters

T4 = "Var15";
fs = 256;
annotator_A = readtable("annotations_2017_A_fixed.csv");

%% Feature Extraction

X = [];
Y = [];
patient = [];

for patient_number = [7,11,14,15,16,17,21,22,31,34,41,63,66,76,78,79,13,36,44,47,52,62,67,69,71]
    [fv, N]= featureextraction(eval(['filtering(extract("D:\eeg_data\eeg' num2str(patient_number) '_data.txt",T4),fs)']),1,fs);
    % Concatenate all patient feature matrices and annotation vectors in X
    % and Y
    X = [X; fv];
    Y = [Y; annotator(annotator_A,patient_number)']; 
    patient = [patient; patient_number*ones(N,1)];
end

%% SVM

X = normalize(X, 2, "zscore");
stats = [];

for j = [7,11,14,15,16,17,21,22,31,34,41,63,66,76,78,79,13,36,44,47,52,62,67,69,71]
    indices = find(patient ~= j);
    
    % Balancing so equal amount of seizure and non-seizure epochs
    count= sum(Y(indices,:));
    one_indices = find(Y(indices,:)==1);
    zero_indices = find(Y(indices,:)==0);
    indx = randsample(zero_indices,round(count));
    indx = [indx; one_indices]; 

    % Train the Network
    SVMmodel= fitcsvm(X(indices(indx),:), Y(indices(indx),:), 'KernelFunction','RBF','KernelScale','auto');
    
    % Test the Network
    YPred = predict(SVMmodel,X(find(patient==j),:));
    
    % Performance metrics       
    FP= sum((YPred==1)&(Y(find(patient==j))'==0));
    TP= sum((YPred==1)&(Y(find(patient==j))'==1));
    FN= sum((YPred==0)&(Y(find(patient==j))'==1));
    TN= sum((YPred==0)&(Y(find(patient==j))'==0));
    FPR=FP/(FP+TN);
    FNR=FN/(FN+TP);
    sensitivity = 1-FNR;
    specificity = 1-FPR;
    BA = (sensitivity+specificity)/2;

    stats = [stats; BA, FPR, FNR];
end

mean(stats)    

%% Cos KNN

k_neighbours = 5;
stats = [];

for j = [7,11,14,15,16,17,21,22,31,34,41,63,66,76,78,79,13,36,44,47,52,62,67,69,71]
    indices = find(patient ~= j);
    
    % Balancing so equal amount of seizure and non-seizure epochs
    count= sum(Y(indices,:));
    one_indices = find(Y(indices,:)==1);
    zero_indices = find(Y(indices,:)==0);
    indx = randsample(zero_indices,round(count));
    indx = [indx; one_indices]; 

    train = X(indices(indx),:);
    labels = Y(indices(indx));
    test = X(find(patient==j),:);
    YPred = zeros(size(test,1),1);

    for l= 1:size(test,1)
        %distances = sqrt(sum(((train-test(l,:)).^2),2)); % euclidean   
        distances = (train * test(l,:)') ./ (sqrt(sum(train.^2, 2)) * norm(test(l,:))); % cosine
        [~, indices] = sort(distances,'descend');
        neighbours = labels(indices(1:k_neighbours));
        YPred(l) = mode(neighbours);
    end
    
    % Test the Network       
    FP= sum((YPred==1)&(Y(find(patient==j))'==0));
    TP= sum((YPred==1)&(Y(find(patient==j))'==1));
    FN= sum((YPred==0)&(Y(find(patient==j))'==1));
    TN= sum((YPred==0)&(Y(find(patient==j))'==0));
    FPR=FP/(FP+TN);
    FNR=FN/(FN+TP);
    sensitivity = 1-FNR;
    specificity = 1-FPR;
    BA = (sensitivity+specificity)/2;

    stats = [stats; BA, FPR, FNR];
end
mean(stats)   

%% PRNN

X = normalize(X, 2, "zscore");
stats = [];

for j = [7,11,14,15,16,17,21,22,31,34,41,63,66,76,78,79,13,36,44,47,52,62,67,69,71]
    indices = find(patient ~= j);
    
    % Balancing so equal amount of seizure and non-seizure epochs
    count= sum(Y(indices,:));
    one_indices = find(Y(indices,:)==1);
    zero_indices = find(Y(indices,:)==0);
    indx = randsample(zero_indices,round(count));
    indx = [indx; one_indices];
    
    % Create the feedforward neural network
    hidden_neurons = 20; 
    net = patternnet(hidden_neurons);
    net.performFcn = 'crossentropy'; 
    view(net)

    % Train the Network
    [net,tr] = train(net,X(indices(indx),:).',Y(indices(indx),:).');

    % Test the Network
    YPred = net(X(find(patient==j),:)');
    
    % Test the Network    
    Ypred = double(YPred > 0.50);    
    FP= sum((Ypred==1)&(Y(find(patient==j))'==0));
    TP= sum((Ypred==1)&(Y(find(patient==j))'==1));
    FN= sum((Ypred==0)&(Y(find(patient==j))'==1));
    TN= sum((Ypred==0)&(Y(find(patient==j))'==0));
    FPR=FP/(FP+TN);
    FNR=FN/(FN+TP);
    sensitivity = 1-FNR;
    specificity = 1-FPR;
    BA = (sensitivity+specificity)/2;

    stats = [stats; BA, FPR, FNR];
end
mean(stats)    

%% Functions

function [vector,N] = featureextraction(data, epoch,fs)
    sizeofdata= size(data);
    sample_per_window= epoch*fs;
    n_windows = length(1:sample_per_window:sizeofdata(1,2)-sample_per_window+1);
    % Initialise feature vector with zeros
    vector = zeros(n_windows,20);
    wavelet = 'dB4';
    levels = 5;
    index = 1;
    for l= 1:sample_per_window:sizeofdata(1,2)-sample_per_window+1
        % Extract window
        datawindow= data(l:l+ sample_per_window-1);
        % Perform Daubechies wavelet decomposition using 5 levels
        [c,L] = wavedec(datawindow,levels,wavelet);
        % Initialise feature vector for each window
        vector_epoch = [];
        % 1D approximation coefficients (Delta) feature extraction
        approximation = appcoef(c,L, wavelet,levels); 
        vector_epoch = [vector_epoch, mean(approximation), std(approximation), kurtosis(approximation), skewness(approximation)];
        % Detail at levels 3-4-5 (Theta-Alpha-Beta) feature extraction
        for level= 3:levels
            coeff = detcoef(c,L,level);
            vector_epoch = [vector_epoch, mean(coeff), std(coeff), kurtosis(coeff), skewness(coeff)];
        end
        % Detail at level 1 and 2 (Gamma) feature extraction
        gamma_coeffs = [detcoef(c,L,1),detcoef(c,L,2)];
        vector_epoch = [vector_epoch, mean(gamma_coeffs), std(gamma_coeffs), kurtosis(gamma_coeffs), skewness(gamma_coeffs)];
        % Concatenate feature vectors for all windows
        vector(index,:)= vector_epoch;
        index = index + 1;
    end
    N= n_windows;
end

% Each level corresponds to these frequency bands:
    % Approximation at level 5: 0 to fs/64 Hz 0-4Hz
    % Detail at level 5: fs/64 to fs/32 Hz 4-8Hz
    % Detail at level 4: fs/32 to fs/16 Hz 8-16Hz
    % Detail at level 3: fs/16 to fs/8 Hz 16-32Hz
    % Detail at level 2: fs/8 to fs/4 Hz 32-64Hz
    % Detail at level 1: fs/4 to fs/2 Hz 64-128Hz

function filtered_data = filtering(data,fs)
    % 4th order Butterworth bandpass (1-100Hz) and bandstop (45-55Hz)
    [b,a] = butter(4,[1 100]*2/fs,'bandpass');
    [b2,a2] = butter(4,[45 55]*2/fs,'stop');
    filtered_data = filter(b2, a2, filter(b,a,data));
end


function annotator_data = annotator(data,patient)
    annotator_data = table2array(data(2:end,patient))';
    annotator_data = annotator_data(~isnan(annotator_data));
end

function data = extract(path,channels)
    if nargin < 2
        channels = ["Var2", "Var3", "Var4", "Var5", "Var6", "Var7", "Var8", "Var9", "Var10", "Var11", "Var12", "Var13", "Var14", "Var15", "Var16", "Var17", "Var18", "Var19", "Var20"]; 
    end
    opts = detectImportOptions(path);
    opts.SelectedVariableNames = channels;
    data = readtable(path,opts);
    data = table2array(data)';
    data = data(:,2:end);
end