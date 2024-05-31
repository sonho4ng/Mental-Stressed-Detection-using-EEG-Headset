# This file use to record data.
import numpy as np
import pickle
import serial
from PIL import Image, ImageTk
import tkinter as tk
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
if serial.Serial:
    serial.Serial().close()

import sys
import os
import tensorflow as tf

# Get the current working directory (which should be the directory containing the Jupyter notebook)
current_dir = os.getcwd()

# Construct the absolute path to the DataPreprocessing directory
data_preprocessing_path = os.path.join(current_dir, '..', 'DataPreprocessing')

# Add this path to sys.path
sys.path.append(os.path.abspath(data_preprocessing_path))
from Preprocessing import slide_func, filter_data, FeatureExtract, create_feature_matrix, create_stft_matrix, time_series_features, hjorth_features, fractal_features


# Open the serial port
# s = serial.Serial("/dev/tty.usbmodem1301", baudrate=57600)  # COMx in window or /dev/ttyACMx in Ubuntu with x is number of serial port.
loaded_model_1 = pickle.load(open("../trained_model/StackingClassifier_Enhanced.h5", 'rb'))
loaded_model_2 = pickle.load(open("../trained_model/RandomForestClassifier_GLCM.h5", 'rb'))
loaded_model_3 = pickle.load(open("../trained_model/KNN.h5", 'rb'))
loaded_model_4 = pickle.load(open("../trained_model/ANN.h5", 'rb'))
loaded_model_5 = pickle.load(open("../trained_model/DecisionTreeClassifier_GLCM.h5", 'rb'))
loaded_model_6 = pickle.load(open("../trained_model/KNeighborsClassifier_GLCM.h5", 'rb'))
loaded_model_7 = pickle.load(open("../trained_model/GradientBoosting.h5", 'rb'))
loaded_model_8 = pickle.load(open("../trained_model/MLPClassifier.h5", 'rb'))
loaded_model_9 = pickle.load(open("../trained_model/SVC_GLCM.h5", 'rb'))
loaded_model_10 = pickle.load(open("../trained_model/SVMClassifier.h5", 'rb'))
print("START!")

reverse_label_mapping = {-1: 'Stressed', 0: 'Relaxed', 1: 'Goodmood'}
scaler1 = pickle.load(open("../trained_model/TimeFreqscaler.pkl", 'rb'))
scaler2 = pickle.load(open("../trained_model/Timescaler.pkl", 'rb'))
def Freq_feature(slide):
    test_slide = pd.DataFrame.from_dict(FeatureExtract(slide, plot=0)).values
    # test_slide = scaler.fit_transform(test_slide)
    # print(test_slide.shape)
    return test_slide

def TimeFreq_feature(slide):
    print(slide.shape)
    stft_matrix = create_stft_matrix(slide)
    print(stft_matrix.shape)
    test_slide = create_feature_matrix(stft_matrix)
    print(test_slide)
    test_slide = scaler1.transform(test_slide)
    print(test_slide.shape)
    return test_slide

def Time_feature(slide):
    features = np.empty((0,7))
    feature1 = time_series_features(slide)
    feature2 = hjorth_features(slide)
    feature3 = fractal_features(slide)
    #feature4 = entropy_features(i)
    feature = np.hstack(( feature1, feature2, feature3))
    features = np.vstack((features, feature))
    features = scaler2.transform(features)
    return features

def ANN_feature(feature):
    # ann_feature = np.squeeze(np.expand_dims(feature, axis = 0))
    ann_feature = np.expand_dims(feature, axis = 0)
    # batch_size = 32
    # data_new = tf.data.Dataset.from_tensor_slices(feature).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return ann_feature

def status(slide):
    unique_elements, counts = np.unique(slide, return_counts=True)
    max_count_index = np.argmax(counts)
    most_frequent_element = int(unique_elements[max_count_index])
    status = reverse_label_mapping[most_frequent_element]
    return status

def status_ann(predictions):
    label = np.argmax(predictions)
    status = reverse_label_mapping[label-1]
    return status



def start_test():
    global status_label 
    x = 0
    y = np.array([], dtype=int)
    k = 15  # window
    sample_rate = 512
    window_size = k * sample_rate
    feature = []
    path = "../CollectedData/new_data/ThaiBuon.txt"
    path = "/Users/nguyentrithanh/Documents/Lab/EEG_SangTaoTre-main/Data_and_AI\ThanhBuon.txt"
    file = open(path, "r")
    while x < (1000 * sample_rate):
        if x % sample_rate == 0:
            print(x // sample_rate)
        # data = file.readline().decode('utf-8').rstrip("\r\n")
        data = file.readline()
        y = np.append(y, int(data))
        x += 1
        if x >= window_size:
            if x % (1 * sample_rate) == 0:
                sliding_window_start = x - window_size
                sliding_window_end = x
                slide = np.array(y[sliding_window_start:sliding_window_end])  # sliding_window ~ ys
                feature_window = FeatureExtract(slide, plot=1)
                feature.append(feature_window)

                freq_feature = Freq_feature(slide)
                timefreq_feature = TimeFreq_feature(slide)
                time_feature = Time_feature(slide)
                # print("Time: ", time_feature)
                # print("Timefreq: ", timefreq_feature)
                print("Freq: ", freq_feature.shape)
                predictions_1 = loaded_model_1.predict(freq_feature)
                status_1 = status(predictions_1)


                predictions_2 = loaded_model_2.predict(timefreq_feature)
                status_2 = status(predictions_2)

                predictions_3 = loaded_model_3.predict(time_feature)
                status_3 = status(predictions_3)

                # test = np.expand_dims(freq_feature, axis = 0)
                # print("bug: ", test[0].shape)

                predictions_4 = loaded_model_4.predict(ANN_feature(freq_feature))
                status_4 = status_ann(predictions_4)

                predictions_5 = loaded_model_5.predict(timefreq_feature)
                status_5 = status(predictions_5)

                predictions_6 = loaded_model_6.predict(timefreq_feature)
                status_6 = status(predictions_6)

                predictions_9 = loaded_model_9.predict(timefreq_feature)
                status_9 = status(predictions_9)

                predictions_7 = loaded_model_7.predict(freq_feature)
                status_7 = status(predictions_7)

                predictions_8 = loaded_model_8.predict(freq_feature)
                status_8 = status(predictions_8)

                predictions_10 = loaded_model_10.predict(freq_feature)
                status_10 = status(predictions_10)



                if status_label is None:
                    status_label_1 = tk.Label(window, text="StackingClassifier_Enhanced: ", font=("Arial", 14))
                    status_label_1.pack(pady=10)
                    status_label_2 = tk.Label(window, text="RandomForestClassifier_GLCM: ", font=("Arial", 14))
                    status_label_2.pack(pady=10)
                    status_label_3 = tk.Label(window, text="KNN: ", font=("Arial", 14))
                    status_label_3.pack(pady=10)
                    status_label_4 = tk.Label(window, text="ANN: ", font=("Arial", 14))
                    status_label_4.pack(pady=10)
                    status_label_5 = tk.Label(window, text="DecisionTreeClassifier_GLCM: ", font=("Arial", 14))
                    status_label_5.pack(pady=10)
                    status_label_6 = tk.Label(window, text="KNeighborsClassifier_GLCM: ", font=("Arial", 14))
                    status_label_6.pack(pady=10)
                    status_label_7 = tk.Label(window, text="GradientBoosting: ", font=("Arial", 14))
                    status_label_7.pack(pady=10)
                    status_label_8 = tk.Label(window, text="MLPClassifier: ", font=("Arial", 14))
                    status_label_8.pack(pady=10)
                    status_label_9 = tk.Label(window, text="SVC_GLCM: ", font=("Arial", 14))
                    status_label_9.pack(pady=10)
                    status_label_10 = tk.Label(window, text="SVMClassifier: ", font=("Arial", 14))
                    status_label_10.pack(pady=10)
                    status_label = 1
                status_label_1.config(text=f"StackingClassifier_Enhanced: {status_1}")
                status_label_2.config(text=f"RandomForestClassifier_GLCM: {status_2}")
                status_label_3.config(text=f"KNN: {status_3}")
                status_label_4.config(text=f"ANN: {status_4}")
                status_label_5.config(text=f"DecisionTreeClassifier_GLCM: {status_5}")
                status_label_6.config(text=f"KNeighborsClassifier_GLCM: {status_6}")
                status_label_7.config(text=f"GradientBoosting: {status_7}")
                status_label_8.config(text=f"MLPClassifier: {status_8}")
                status_label_9.config(text=f"SVC_GLCM: {status_9}")
                status_label_10.config(text=f"SVMClassifier: {status_10}")
                
            # print(int(data))
            # file.write(data)
            # file.write('\n')
                show_image()
                time.sleep(0.1)
    # np.savetxt("D:\Tuda\Research\Data_and_AI\Data_new\Tuda_00000.txt", y, fmt="%d")  # Save in int


# Use to get data txt


image_path1 = 'test.png'

status_label = None

def show_image():
    image1 = Image.open(image_path1)
    photo1 = ImageTk.PhotoImage(image1)
    image_frame1.configure(image=photo1)
    window.update()
    image1.close()


window = tk.Tk()

window.title("Awake Drive")

isPause = tk.IntVar()
pause_button = tk.Checkbutton(window, text="Dừng", variable=isPause, onvalue=1, offvalue=0)
pause_button.pack(pady=10)

start_button = tk.Button(window, text="Bắt đầu", command=start_test)
start_button.pack(pady=10)
# Raw wave
image_frame1 = tk.Label(window, width=1000, height=500, bg="white")
image_frame1.pack(side=tk.LEFT)

show_image()
# Tạo một khung văn bản để hiển thị trạng thái buồn ngủ/tỉnh táo


window.mainloop()
# Close the serial port
print("DONE")
# s.close()
# file.close()
