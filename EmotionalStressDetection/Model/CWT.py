import tensorflow as tf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pywt
import keras
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from sklearn.preprocessing import Normalizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import pickle
np.random.seed(1)

# Datas
files = {("BachBuon.txt", 0), ("BachCalm.txt", 1), ("QuangCalm3.txt", 1), ("QuangVui3.txt", 2), ("SonBuon3.txt", 0), ("SonBuon4.txt", 0), ("SonVui4.txt", 2), ("ThaiBuon.txt", 0), ("ThaiCalm.txt", 1), ("ThaiCalm2.txt", 1), ("ThaiVui.txt", 2), ("ThaiVui2.txt", 2), ("ThanhfBuon.txt", 0), ("ThanhfCalm.txt", 1), ("ThanhfCalm2.txt", 1), ("ThanhfVui.txt", 2), ("ThanhfVui2.txt", 2)}


# Preprocessing
def file_signals(processed_file: list, window_size: int):
    ''' Collecting signals from a file by window sliding'''
    end = int(((len(processed_file)/512) - window_size) + 1)
    state_signals = [processed_file[(512*i):(512*(window_size + i))] for i in range(end)]
    return state_signals
    
def filter(data):
    ''' Band-pass filter and interpolation '''
    band = [0.5 / (0.5 * 512), 40 / (0.5 * 512)]
    b, a = sp.signal.butter(4, band, btype='band', analog=False, output='ba')
    data = sp.signal.lfilter(b, a, data)

    filtered_data = data[(np.abs(data) <= 256)]
    x = np.arange(len(filtered_data))
    interpolated_data = interp1d(x, filtered_data)(np.linspace(0, len(filtered_data) - 1, len(data)))
    return interpolated_data

# Extracting data
datas, classes = [], []

for filename, label in files:
    raw_data = np.loadtxt("../CollectedData/new_data/" + filename)
    filtered_data = filter(raw_data)
    classes.append(label)
    datas.append(filtered_data)

# Standardization
    # Each member of the standardized data is a filtered data taken from one file.
scaler = Normalizer()
scaled_data = scaler.fit_transform(np.array(datas))
standardized_data = list(scaled_data)

# Collecting signals
signals, labels = [], []
for i in range(len(standardized_data)):
    signals_by_window = file_signals(standardized_data[i], 5)
    signals.extend(signals_by_window)
    labels.extend([classes[i] for j in range(len(signals_by_window))])
    
# Randomization
permutation = np.random.permutation(len(signals))
signals = np.array(signals)[permutation]
labels = np.array(labels)[permutation]

# Time-frequency domain extractions using Continuous Wavelet Transform
def tfDomain(signals, scales = range(1, 128), waveletName = 'morl'):
    cwt_data = np.ndarray(shape = (len(signals), len(scales), len(scales)))
    for i in range(len(signals)):
        coef, freq = pywt.cwt(signals[i], scales, wavelet = waveletName)
        coef_ = coef[:, :(len(scales))]
        cwt_data[i, :, :] = coef_
    return cwt_data

features = tfDomain(signals)

# Statistics + Visualization
def statistics(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis = 1)
    y_test = np.argmax(y_test, axis = 1)
    print(classification_report(y_test, y_pred, labels = [0, 1, 2]))
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels = ["Stressed", "Calm", "Happy"])
    disp.plot()
    plt.show()
    return

def performance(history):
    ''' Plot the model's accuracy and loss on the training and validation datasets. '''
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()
    return

# Building the CNN model
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size = 0.1,
    shuffle = True, stratify = labels, random_state=1
)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, 3)
y_test = keras.utils.to_categorical(y_test, 3)

model = Sequential(
    [
     Conv1D(32, kernel_size = 2, activation = 'relu', input_shape = (127, 127)),
     MaxPooling1D(pool_size = 2), 
     Dropout(0.5),
     Conv1D(64, kernel_size = 2, activation = 'relu'), 
     MaxPooling1D(pool_size = 2),
     Dropout(0.5)
     ]
)

model.add(Flatten())
model.add(Dense(1000, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)))
model.add(Dense(3, activation= "softmax"))

model.compile(loss = keras.losses.categorical_crossentropy, 
              optimizer = keras.optimizers.Adam(learning_rate= 0.001), 
              metrics = ['acc'])

early_stopping = EarlyStopping(monitor = 'val_loss', patience= 20, restore_best_weights= True)

history = model.fit(x_train, y_train, batch_size = 32, epochs = 100, verbose = 1, validation_data = (x_test, y_test), callbacks = [early_stopping])

statistics(model, x_test, y_test)

train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose = 0)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f'Training Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')

performance(history)
# model.save("CNN.h5")

filename = f'../trained_model/CWT.h5'
pickle.dump(model, open(filename, 'wb'))
with open('../trained_model/CWTscaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)