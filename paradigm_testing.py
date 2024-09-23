import sounddevice as sd
import argparse
import time
import numpy as np
from scipy.signal import chirp
import tkinter as tk
from PIL import Image, ImageTk
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
import pandas as pd
import matplotlib.pyplot as plt
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes, WaveletTypes, NoiseEstimationLevelTypes, \
    WaveletExtensionTypes, ThresholdTypes, WaveletDenoisingTypes, WindowOperations, DetrendOperations
import os
from scipy import stats

def main():

    # Define session name- it must only include letters, numbers and underscores NO spaces or dashes
    session_name = '23_02_24_josh3'

    # Connect these channels to the board
    ear1 = 1
    ear2 = 2
    forehead = 3
    temporal = 4
    occipital = 5
    # Connect ipsilateral mastoid electrode to BIAS and contralateral mastoid electrode to SRB

    # Creates folder for the session in the same path as this python file
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, session_name)
    if not os.path.exists(session_name):
        os.makedirs(session_name)


    # region Setting parameters of the board

    params = BrainFlowInputParams()
    parser = argparse.ArgumentParser(description='Connect to a Ganglion board using BrainFlow.')
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=30)
    parser.add_argument('--serial-port', type=str, help='serial port for Windows connection', required=False,
                        default='COM4')
    parser.add_argument('--board-id', type=int, help='board id, should be set to Ganglion\'s board id', required=False,
                        default=BoardIds.CYTON_BOARD.value)
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--file', type=str, help='file for playback or logging', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    args = parser.parse_args()
    params = BrainFlowInputParams()
    params.serial_port = args.serial_port
    params.other_info = args.other_info
    params.mac_address = args.mac_address
    params.serial_number = args.serial_number
    params.timeout = args.timeout
    params.file = args.file
    board = BoardShim(args.board_id, params)
    nfft = DataFilter.get_nearest_power_of_two(BoardShim.get_sampling_rate(args.board_id))
    eeg_channels = BoardShim.get_eeg_channels(args.board_id)
    marker_channel = BoardShim.get_marker_channel(args.board_id)
    fs = BoardShim.get_sampling_rate(args.board_id)
    print('EEG channel index: ',eeg_channels,'\nMarker channel index: ',marker_channel)

    # endregion

    # region Functions: write, filter, read, restore, create tkinter window, update window
    def write_data(raw_data, raw_file_name, session):
        # Save raw data as csv file in session folder
        file_path = os.path.join(session,f'{raw_file_name}_{session}.csv')
        DataFilter.write_file(raw_data, file_path, 'w')

    def filter_data(raw_file_name, filtered_file_name, session):
        # Open raw data
        data =  DataFilter.read_file(os.path.join(session,f'{raw_file_name}_{session}.csv'))

        # Filter data
        for channel in eeg_channels:
            # Butterworth bandpass filter from 5 to 50 Hz of order 4
            DataFilter.perform_bandpass(data[channel], BoardShim.get_sampling_rate(args.board_id), 5.0, 50.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            # Butterworth bandstop filter from 50 to 60 Hz of order 4
            DataFilter.perform_bandstop(data[channel], BoardShim.get_sampling_rate(args.board_id), 50.0, 60.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)

        # Denoising with moving average filter (Window = 12ms)
        for channel in eeg_channels:
            DataFilter.perform_rolling_filter(data[channel], 3, AggOperations.MEAN.value)

        # Denoising with moving median filter (Window = 12ms)
        for channel in eeg_channels:
            DataFilter.perform_rolling_filter(data[channel], 3, AggOperations.MEDIAN.value)

        # Denoising with wavelet transform
        for channel in eeg_channels:
            DataFilter.perform_wavelet_denoising(data[channel], WaveletTypes.BIOR3_9, 3,
                                                 WaveletDenoisingTypes.SURESHRINK, ThresholdTypes.HARD,
                                                 WaveletExtensionTypes.SYMMETRIC, NoiseEstimationLevelTypes.FIRST_LEVEL)

        # Save filtered data as csv
        file_path = os.path.join(session, f'{filtered_file_name}_{session}.csv')
        DataFilter.write_file(data, file_path, 'w')

    def read(test, session):
        # Open filtered csv data file
        read_data = DataFilter.read_file(os.path.join(session, f'data_{test}_filtered_{session}.csv'))
        return read_data

    def restore(test, session):
        # Restore filtered data as pandas format
        restored_data = DataFilter.read_file(os.path.join(session,f'data_{test}_filtered_{session}.csv'))
        restored_data = pd.DataFrame(np.transpose(restored_data))
        return restored_data

    def create_tk():
        # Generate tkinter root
        root = tk.Tk()
        root.state('zoomed')

        # Create tkinter images
        oddball = ImageTk.PhotoImage(
            Image.open(r"C:\Users\rmexi\OneDrive - Imperial College London\Y4\FYP\Validation\oddball.jpg").resize((1400,1000)))
        alpha = ImageTk.PhotoImage(
            Image.open(r"C:\Users\rmexi\OneDrive - Imperial College London\Y4\FYP\Validation\alpha.jpg").resize((1400,1000)))
        assr = ImageTk.PhotoImage(
            Image.open(r"C:\Users\rmexi\OneDrive - Imperial College London\Y4\FYP\Validation\assr.jpg").resize((1400,1000)))
        check1 = ImageTk.PhotoImage(
            Image.open(r"C:\Users\rmexi\OneDrive - Imperial College London\Y4\FYP\Validation\check1.jpg").resize((1400,1000)))
        check2 = ImageTk.PhotoImage(
            Image.open(r"C:\Users\rmexi\OneDrive - Imperial College London\Y4\FYP\Validation\check2.jpg").resize((1400,1000)))
        ssvep = ImageTk.PhotoImage(
            Image.open(r"C:\Users\rmexi\OneDrive - Imperial College London\Y4\FYP\Validation\ssvep.jpg").resize((1400,1000)))

        return root, oddball, alpha, assr, check1, check2, ssvep

    def update_screen(new_screen):
        # Update window image
        label.config(image=new_screen)

    # endregion

    # region Session
    board.prepare_session()

    # region Auditory Oddball

    # Open tkinter window, display oddball instructions for 15 seconds and close
    root, oddball, alpha, assr, check1, check2, ssvep  = create_tk()
    label = tk.Label(root)
    label.pack()
    update_screen(oddball)
    root.after(15000, root.destroy)
    root.mainloop()

    # Set the sampling frequency to 44.1kHz for audio quality
    Fs = 44100
    Ts = 1 / Fs

    # Frequencies of tones and their probabilities
    freq = [1000, 500]
    P = [0.1, 0.9]

    # Generate random sequence of tones
    tone_number = 150
    tone_sequence = []

    for i in range(tone_number):
        f = np.random.choice(freq, p=P)
        tone_sequence.append(f)

    # Function to play a tone
    def play_tone(freq, duration):
        t = np.arange(0, duration, Ts)
        waveform = np.sin(2 * np.pi * freq * t)
        sd.play(waveform, Fs)
        sd.wait()

    # Play 10 standard tones
    for i in range(10): #10
        play_tone(500, 0.3)
        time.sleep(0.3)

    # Play a long beep
    play_tone(500, 2)

    # Start stream and test
    board.start_stream()
    time.sleep(2) # Avoid initialisation voltage
    for f in tone_sequence:
        board.insert_marker(f) # insert marker at each stimuli
        play_tone(f, 0.3)
        time.sleep(0.3) # Overall time elapsed for each tone is 0.6s

    time.sleep(1)

    # Collect data, save as csv, stop stream and save filtered data
    data_p300 = board.get_board_data()
    write_data(data_p300, 'data_p300_raw', session_name)
    board.stop_stream()
    filter_data('data_p300_raw', 'data_p300_filtered',session_name)

    time.sleep(10)

    # endregion

    # region ASSR

    # Open tkinter window, display ASSR instructions for 15 seconds and close
    root, oddball, alpha, assr, check1, check2, ssvep = create_tk()
    label = tk.Label(root)
    label.pack()
    update_screen(assr)
    root.after(15000, root.destroy)
    root.mainloop()

    # Function to play a modulated chirp
    def play_chirp(f_start, f_end, duration, total_t, modulation):
        t = np.arange(0, duration, Ts)
        t2 = np.arange(0, total_t, Ts)
        rep = int(total_t / duration)
        waveform = chirp(t, f0=f_start, f1=f_end, t1=duration, method='logarithmic')
        full_waveform = np.tile(waveform, rep)
        mod = 0.5 * np.sin(2 * np.pi * modulation * t2) + 0.5
        modulated = full_waveform * mod
        sd.play(modulated, Fs)
        sd.wait()

    # Start stream and test
    board.start_stream()
    time.sleep(2)  # Avoid initialisation voltage
    play_chirp(500, 4000, 0.01, 90, 40)

    # Collect data, stop stream and save as csv, filter and save
    data_assr = board.get_board_data()
    board.stop_stream()
    write_data(data_assr, 'data_assr_raw', session_name)
    filter_data('data_assr_raw','data_assr_filtered',session_name)

    time.sleep(10)

    # endregion

    # region Alpha Modulation

    # Open tkinter window, display alpha modulation instructions for 15 seconds and close
    root, oddball, alpha, assr, check1, check2, ssvep = create_tk()
    label = tk.Label(root)
    label.pack()
    update_screen(alpha)
    root.after(15000, root.destroy)
    root.mainloop()

    # Play tone to indicate start of eyes closed
    play_tone(500, 2)

    # Start 5 minute test and stream
    board.start_stream()
    time.sleep(300) #300

    # Collect data, stop stream and save as csv, filter data
    data_ec = board.get_board_data()
    write_data(data_ec, 'data_ec_raw', session_name)
    board.stop_stream()
    filter_data('data_ec_raw','data_ec_filtered',session_name)

    # Play tone to indicate start of eyes open
    play_tone(500, 2)

    # Start 5 minute test and stream
    board.start_stream()
    time.sleep(300)

    # Collect data, stop stream and save as csv, filter data
    data_eo = board.get_board_data()
    write_data(data_eo, 'data_eo_raw', session_name)
    board.stop_stream()
    filter_data('data_eo_raw','data_eo_filtered',session_name)

    # Play tone to indicate end of test
    play_tone(500, 2)

    time.sleep(10)

    # endregion

    # region SSVEP

    # Open tkinter window, display SSVEP instructions for 15 seconds and close
    root, oddball, alpha, assr, check1, check2, ssvep = create_tk()
    label = tk.Label(root)
    label.pack()
    update_screen(ssvep)
    root.after(15000, root.destroy)
    root.mainloop()

    data_ssvep = [] # Pre-allocate array

    # Function to update the image displayed
    def update_image():
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Calculate the current frame - 10Hz flicker = 1 every 0.1 seconds
        current_frame = int(elapsed_time / 0.1) % 2

        # Update the label
        if current_frame == 0:
            label.config(image=check1)
        else:
            label.config(image=check2)

        # Next update in 0.1s
        root.after(10, update_image)

    # Do test 5 times with 10s rest
    for i in range(5):
        # Start the flickering effect and the stream
        start_time = time.time()
        board.start_stream()
        time.sleep(2) # Avoid initialisation voltage
        root, oddball, alpha, assr, check1, check2, ssvep = create_tk()
        label = tk.Label(root)
        label.pack()
        update_image()
        root.after(10000, root.destroy) # 10 second flickering effect
        root.mainloop()
        data_ssvep.append(board.get_board_data()) # collect data
        board.stop_stream() # stop test
        time.sleep(10)  # 10 second rest

    # Concatenate all 5 repeats into one data array and save as csv, filter data and save
    min_size = min(arr.shape[1] for arr in data_ssvep)
    data_ssvep = [arr[:, :min_size] for arr in data_ssvep]
    data_ssvep = np.hstack(data_ssvep)
    write_data(data_ssvep, 'data_ssvep_raw', session_name)
    filter_data('data_ssvep_raw','data_ssvep_filtered',session_name)

    # endregion

    # Stop session
    board.release_session()

    play_tone(500, 2) # end of testing

    # endregion

    # region Processing

    # Function: PSD and Bandpower extraction
    def bandpower(data, channel1, channel2):
        # Average both in-ear channels for smoothening
        av_channel = (data[channel1] + data[channel2])/2

        # PSD returns (uV)^2/Hz
        psd = DataFilter.get_psd_welch(av_channel, nfft, nfft // 2, BoardShim.get_sampling_rate(args.board_id),WindowOperations.BLACKMAN_HARRIS.value)

        # Band Power
        delta = DataFilter.get_band_power(psd,0, 5)
        theta = DataFilter.get_band_power(psd,  5, 9)
        alpha = DataFilter.get_band_power(psd, 9, 14)
        beta = DataFilter.get_band_power(psd,  14, 33)
        gamma = DataFilter.get_band_power(psd,  33, 100)
        band_power = [delta, theta, alpha, beta, gamma]
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        psd_vals = psd[1][:100]
        freqs_vals = psd[0][:100]

        return band_power, bands, psd_vals, freqs_vals

    # Function: P300 analysis
    def plot_p300(data, channel1, channel2):
        # Average both in-ear channels for smoothening
        av_channel = (data[channel1] + data[channel2])/2

        fs = BoardShim.get_sampling_rate(args.board_id)
        t = (np.arange(0, len(data[channel1])))/fs # time-axis for whole test
        t_resp = np.linspace(0, 0.7, 176) # time-axis for 0.7s epochs

        normal_responses = []
        deviant_responses = []

        # Section data in 0.7s epochs at every stimuli
        for i in range(len(data[marker_channel])):
            if int(data[marker_channel][i]) == 500: # Normal tone
                baseline = np.mean(av_channel[int(i - 0.1 * fs):i + 1]) # Baseline correction 0.1s pre-stimulus
                corrected = av_channel[int(i - 0.1 * fs):int(i + 1 + 0.6 * fs)] - baseline # Epoch from 0.1s pre-stim to 0.6s post-stim
                corrected = corrected.to_numpy()
                normal_responses.append(corrected)
            elif int(data[marker_channel][i]) == 1000: # Deviant tone
                baseline = np.mean(av_channel[int(i - 0.1 * fs):i + 1]) # Baseline correction 0.1s pre-stimulus
                corrected = av_channel[int(i - 0.1 * fs):int(i + 1 + 0.6 * fs)] - baseline # Epoch from 0.1s pre-stim to 0.6s post-stim
                corrected = corrected.to_numpy()
                deviant_responses.append(corrected)

        normal_responses = np.stack(normal_responses)
        deviant_responses = np.stack(deviant_responses)

        # Calculate average responses
        mean_normal = np.mean(normal_responses, axis=0)
        mean_deviant = np.mean(deviant_responses, axis=0)
        return av_channel, t, t_resp, mean_deviant, mean_normal

    # Open Auditory Oddball data and plot on time domain
    filt_p300 = restore('p300', session_name)
    av_channel, t, t_resp, mean_deviant, mean_normal = plot_p300(filt_p300, ear1,ear2)
    # Plot the whole test
    plt.figure(figsize=(20, 12))
    plt.plot(t,av_channel) # eeg
    plt.plot(t,filt_p300[marker_channel]) # stimuli
    plt.title('Auditory Oddball Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (uV)')
    plt.savefig(os.path.join(session_name, f'p300_{session_name}_graph.jpg')) # Save graph
    plt.show(block=False)
    # Plot mean responses
    plt.figure(figsize=(20, 12))
    plt.plot(t_resp, mean_normal, label='Average normal response')  # normal
    plt.plot(t_resp, mean_deviant, label='Average deviant response', color='red')  # deviant
    plt.title('Auditory Oddball Avergae Responses')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (uV)')
    plt.legend()
    plt.savefig(os.path.join(session_name, f'p300responses_{session_name}_graph.jpg')) # Save graph
    plt.show(block=False)
    # t-test
    mean_normal_poststim = mean_normal[int(0.1*fs):]
    mean_deviant_poststim = mean_deviant[int(0.1 * fs):]
    t, p = stats.ttest_ind(mean_normal_poststim, mean_deviant_poststim)
    if p < 0.05:
        print('Normal and deviant responses are significantly different')
    else:
        print('Normal and deviant responses are NOT significantly different')

    # Open ASSR data and plot the periodogram
    filt_assr = read('assr', session_name)
    bands_assr, bands, psd_assr, freqs_assr = bandpower(filt_assr, ear1,ear2)
    plt.figure(figsize=(20, 12))
    plt.plot(psd_assr, freqs_assr)
    plt.title('ASSR Periodogram')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (uV)^2/Hz')
    plt.savefig(os.path.join(session_name, f'assr_{session_name}_graph.jpg')) # Save graph
    plt.show(block=False)

    # Alpha modulation
    colours = ['blue', 'purple', 'green', 'yellow', 'red']
    # Open Eyes closed data and plot bands
    filt_ec = read('ec', session_name)
    bands_ec, bands, psd_ec, freqs_ec = bandpower(filt_ec, ear1,ear2)
    plt.figure(figsize=(20, 12))
    plt.subplot(1,2,1)
    plt.bar(bands, bands_ec, color=colours)
    plt.title('Eyes Closed Bandpower')
    plt.ylabel('Amplitude (uV)^2/Hz')
    # Open Eyes open data and plot bands
    filt_eo = read('eo', session_name)
    bands_eo, bands, psd_eo, freqs_eo = bandpower(filt_eo, ear1,ear2)
    plt.subplot(1, 2, 2)
    plt.bar(bands, bands_eo, color=colours)
    plt.title('Eyes Open Bandpower')
    plt.ylabel('Amplitude (uV)^2/Hz')
    plt.tight_layout()
    plt.savefig(os.path.join(session_name, f'alpha_{session_name}_graph.jpg')) # Save graph
    plt.show(block=False)
    # dB Drop
    bands_eo = np.array(bands_eo)
    bands_ec = np.array(bands_ec)
    difference = np.absolute(bands_ec - bands_eo)
    dB_drop = np.absolute(20 * np.log10(difference / bands_ec))
    dB_drop_no_alpha = np.concatenate((dB_drop[:2], dB_drop[3:]))
    mean = np.mean(dB_drop_no_alpha)
    std = np.std(dB_drop_no_alpha)
    significance = np.absolute((dB_drop[3] - mean) / std)
    significance = np.ceil(significance * 1000) / 1000
    print(f'The dB drop in alpha is {significance} standard deviations away from the mean dB change')

    # Open SSVEP data and plot periodogram
    filt_ssvep = read('ssvep', session_name)
    bands_ssvep, bands, psd_ssvep, freqs_ssvep = bandpower(filt_ssvep, ear1,ear2)
    plt.figure(figsize=(20, 12))
    plt.plot(psd_ssvep, freqs_ssvep)
    plt.title('SSVEP Periodogram')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (uV)^2/Hz')
    plt.savefig(os.path.join(session_name, f'ssvep_{session_name}_graph.jpg')) # Save graph
    plt.show(block=False)

    # endregion

if __name__ == "__main__":
    main()