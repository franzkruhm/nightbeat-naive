# -*- coding: utf-8 -*-

"""
IMU Heart Rate Estimation from JSON Data
Based on https://github.com/eth-siplab/Nightbeat

This script loads IMU data (x, y, z components) from a JSON file, calculates the magnitude of the signal,
performs STFT analysis to extract the dominant frequency, and estimates heart rate in BPM.  Includes a Butterworth filter for 0.5-3Hz bandpass filtering.

Licensed under GPL3: https://www.gnu.org/licenses/gpl-3.0.html
Author: René Kruhm a.k.a Zanfr
Contact: via GitHub or c0d3@zanfr.fr or code@zanfr.fr
Donations are appreciated at: https://ko-fi.com/zanfr
"""

import json
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

def load_and_process_data(file_path):
    """Loads data from a JSON file and performs initial processing."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    x_values = []
    y_values = []
    z_values = []
    # Iterate through the outer list and then the inner dictionary
    for sample in data[0]:  # Access the first (and only) element of the outer list
        x_values.append(float(sample['x']))
        y_values.append(float(sample['y']))
        z_values.append(float(sample['z']))
    return x_values, y_values, z_values

def calculate_magnitude(x, y, z):
    """Calculates the magnitude of each sample."""
    magnitude = np.sqrt(np.array(x)**2 + np.array(y)**2 + np.array(z)**2)
    return magnitude

def butter_bandpass(cutoff, fs, order=5):
    """Creates a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = cutoff[0] / nyq
    high = cutoff[1] / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, b, a):
    """Applies the Butterworth filter to the data."""
    y = lfilter(b, a, data)
    return y

def stft_analysis(data, fs):
    """Performs STFT analysis and extracts dominant frequency using windowing."""
    N = len(data)
    window = np.hamming(N)  # Apply Hamming window to reduce spectral leakage
    yf = fft(data * window)
    xf = fftfreq(N, 1 / fs)
    positive_frequencies = xf[:N//2]
    magnitudes = np.abs(yf[:N//2])
    # Find the index of the peak frequency (excluding DC component)
    dominant_frequency_index = np.argmax(magnitudes[1:]) + 1  # Exclude DC component (index 0)
    dominant_frequency = positive_frequencies[dominant_frequency_index]
    return dominant_frequency

def calculate_heart_rate(dominant_frequency):
    """Calculates heart rate in BPM from the dominant frequency."""
    heart_rate = dominant_frequency * 60
    return heart_rate

if __name__ == "__main__":
    file_path = "imu.json"  # Replace with your JSON file path
    fs = 100  # Sampling frequency in Hz

    try:
        x, y, z = load_and_process_data(file_path)
        magnitude = calculate_magnitude(x, y, z)

        # Butterworth filter (0.5 - 3 Hz)
        cutoff = [0.5, 3]  # Cutoff frequencies for bandpass filtering
        b, a = butter_bandpass(cutoff, fs, order=5)
        filtered_magnitude = apply_filter(magnitude, b, a)

        # Perform STFT analysis on the filtered magnitude data
        dominant_frequency = stft_analysis(filtered_magnitude, fs)

        # Calculate heart rate from the dominant frequency
        heart_rate = calculate_heart_rate(dominant_frequency)

        print(f"Dominant Frequency: {dominant_frequency:.2f} Hz")
        print(f"Average Heart Rate: {heart_rate/2:.2f} BPM")

        # Plotting (optional)
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_magnitude)  #Plot the filtered data
        plt.title("Magnitude of Filtered IMU Signal")
        plt.xlabel("Sample Index")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
