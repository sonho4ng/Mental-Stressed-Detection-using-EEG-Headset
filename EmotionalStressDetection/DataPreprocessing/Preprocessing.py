import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import mne_features.univariate as mne_f
from scipy.interpolate import interp1d
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
from sklearn.preprocessing import StandardScaler

def slide_func(data, window_size, iter):
    if iter >= window_size:
        if iter % (1 * 512) == 0:
            sliding_window_start = iter - window_size
            sliding_window_end = iter
            sliding_window = np.array(data[sliding_window_start:sliding_window_end])  # sliding_window ~ y
            return sliding_window
        return None
    return None


def filter_data(data):
    # Bandpass filter
    band = [0.5 / (0.5 * 512), 40 / (0.5 * 512)]
    b, a = sp.signal.butter(4, band, btype='band', analog=False, output='ba')
    data = sp.signal.lfilter(b, a, data)

    # plt.hist(data, bins=10, edgecolor='black')
    # filter for EMG by interpolated
    filtered_data = data[(np.abs(data) <= 256)]
    x = np.arange(len(filtered_data))
    interpolated_data = interp1d(x, filtered_data)(np.linspace(0, len(filtered_data) - 1, len(data)))
    return interpolated_data

def split_window(arr):

    # Kích thước của các mảng con và số phần tử trùng lặp
    subarray_size = 15*512
    overlap = 14*512

    # Khởi tạo danh sách để lưu các mảng con
    subarrays = []

    # Sử dụng vòng lặp để tạo các mảng con
    for i in range(0, len(arr) - subarray_size + 1, subarray_size - overlap):
        subarray = arr[i:i + subarray_size]
        subarrays.append(subarray)

    # Chuyển danh sách các mảng con thành mảng NumPy
    subarrays = np.array(subarrays)

    return subarrays

def time_series_features(data):
    
    '''
    Computes the features variance, RMS and peak-to-peak amplitude using the package mne_features.

    Args:
        data (ndarray): EEG data.

    Returns:
        ndarray: Computed features.

    '''

    features = []

    variance = mne_f.compute_variance(data)
    rms = mne_f.compute_rms(data)
    ptp_amp = mne_f.compute_ptp_amp(data)
    # features = np.concatenate((features, [variance, rms, ptp_amp]))
    features.append([variance, rms, ptp_amp])
    features = np.array((features))
    return features

def hjorth_features(data):
    '''
    Computes the features Hjorth mobility (spectral) and Hjorth complexity (spectral) using the package mne_features.

    Args:
        data (ndarray): EEG data.

    Returns:
        ndarray: Computed features.
    '''
    sfreq = 512



    features = []

    mobility_spect = mne_f.compute_hjorth_mobility_spect(sfreq, data)
    complexity_spect = mne_f.compute_hjorth_complexity_spect(sfreq, data)
    # features = np.concatenate((features, [variance, rms, ptp_amp]))
    features.append([mobility_spect, complexity_spect])
    features = np.array((features))
    return features


def higuchi_fd(data, kmin=4, kmax=16, exponent=1):
  """
  Calculates the Higuchi Fractal Dimension (HFD) of a time series, handling potential division by zero and small Avg_di.

  Args:
      data (ndarray): 1-dimensional NumPy array representing the EEG data.
      kmin (int, optional): Minimum value of the box length (default: 4).
      kmax (int, optional): Maximum value of the box length (default: 16).
      exponent (float, optional): Exponent used in the HFD calculation (default: 1).

  Returns:
      float: The calculated Higuchi Fractal Dimension.
  """

  # Input validation (optional)
  # ... (same as before)

  # Pre-compute data length
  N = len(data)

  # Initialize variables
  Dm = 0  # Average fractal dimension
  Lm = []  # List to store Ln(k) values

  # Loop through different box lengths (k)
  for k in range(kmin, kmax + 1):
    Lm.append(np.log(k))  # Calculate Ln(k)

    # Initialize counters for this box length
    Nk = 0
    di = 0

    # Loop through the data with stride k
    for i in range(0, N - k + 1, k):
      # Find the minimum and maximum values within the box
      window = data[i:i+k]
      min_val = np.min(window)
      max_val = np.max(window)

      # Increment counters
      Nk += 1
      di += np.abs(max_val - min_val)

    # Handle zero Nk (if necessary)
    if Nk == 0:
      continue

    # Average normalized distance within each box length
    epsilon = 1e-10  # Small constant to avoid zero division
    Avg_di = di / (k * Nk) + epsilon

    # Calculate Ln(Avg_di)
    Ln_di = np.log(Avg_di)

    # Update Dm using the exponent
    Dm += (Ln_di / Lm[-1])

  # Normalize and return HFD
  HFD = (Dm / (kmax - kmin + 1)) * exponent
  return HFD

def katz_fd(data, scale=1.1):
  """
  Calculates the Katz Fractal Dimension (KFD) of a time series, handling potential division by zero.

  Args:
      data (ndarray): 1-dimensional NumPy array representing the EEG data.
      scale (float, optional): Scaling factor for the distance calculation (default: 1.1).

  Returns:
      float: The calculated Katz Fractal Dimension, or 'nan' if data has constant value.
  """

  # Input validation (optional)
  # ... (same as before)

  # Pre-compute data length
  N = len(data)

  # Initialize variable
  D = 0  # Katz Fractal Dimension

  # Check for constant value data (all elements equal)
  if np.all(data == data[0]):
    return np.nan  # Indicate constant value data

  # Loop through each data point (starting from the second element)
  for i in range(1, N):
    # Calculate distance between consecutive points
    distance = np.abs(data[i] - data[i - 1])

    # Handle zero distance (if necessary)
    if distance == 0:
      # Optionally assign a small non-zero value (e.g., distance = 1e-10)
      continue  # Or implement another handling method

    # Update D based on distance and scaling factor
    D += np.log(scale / distance) / np.log(scale)

  # Normalize and return KFD
  KFD = D / (N - 1)
  return KFD



def fractal_features(data):
    '''
    Computes the Higuchi Fractal Dimension and Katz Fractal Dimension using the package mne_features.

    Args:
        data (ndarray): EEG data.

    Returns:
        ndarray: Computed features.

    '''
    features = []
    
    higuchi = higuchi_fd(data)
    katz = katz_fd(data)
    features.append([higuchi, katz])
    features = np.array((features))
    return features



def entropy_features(data):
    '''
    Computes the features Approximate Entropy, Sample Entropy, Spectral Entropy and SVD entropy using the package mne_features.

    Args:
        data (ndarray): EEG data.

    Returns:
        ndarray: Computed features.

    '''
    sfreq = 512
    features = []
    len = data.shape[0]
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=2)
    data = np.reshape(data, (1, len, 1, 1))

    app_entropy = mne_f.compute_app_entropy(data)
    samp_entropy = mne_f.compute_samp_entropy(data)
    spect_entropy = mne_f.compute_spect_entropy(sfreq, data)
    svd_entropy = mne_f.compute_svd_entropy(data)
    features.append([app_entropy, samp_entropy, spect_entropy, svd_entropy])
    features = np.array((features))
    return features

def FeatureExtract(data, plot):
    f, t, Zxx = sp.signal.stft(data, 512, nperseg=15 * 512, noverlap=14 * 512)
    delta = np.array([], dtype=float)
    theta = np.array([], dtype=float)
    alpha = np.array([], dtype=float)
    beta = np.array([], dtype=float)
    for i in range(0, int(t[-1])):
        indices = np.where((f >= 0.5) & (f <= 4))[0]
        delta = np.append(delta, np.sum(np.abs(Zxx[indices, i])))

        indices = np.where((f >= 4) & (f <= 8))[0]
        theta = np.append(theta, np.sum(np.abs(Zxx[indices, i])))

        indices = np.where((f >= 8) & (f <= 13))[0]
        alpha = np.append(alpha, np.sum(np.abs(Zxx[indices, i])))

        indices = np.where((f >= 13) & (f <= 30))[0]
        beta = np.append(beta, np.sum(np.abs(Zxx[indices, i])))

    abr = alpha / beta
    tbr = theta / beta
    dbr = delta / beta
    tar = theta / alpha
    dar = delta / alpha
    dtabr = (alpha + beta) / (delta + theta)

    diction = {"delta": delta,
               "theta": theta,
               "alpha": alpha,
               "beta": beta,
               "abr": abr,
               "tbr": tbr,
               "dbr": dbr,
               "tar": tar,
               "dar": dar,
               "dtabr": dtabr
               }
    if plot == 1:
        # Tạo hình ảnh chính và các hình ảnh con
        fig = plt.figure(figsize=(10, 5))
        # Plot raw
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(data)
        ax1.set_title('EEG Raw Values')
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('RawValue')
        ax1.set_ylim(-256, 256)

        # Plot STFT
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.pcolormesh(t, f, np.abs(Zxx), vmin=-1, vmax=5, shading='auto')
        ax2.set_title('STFT Magnitude')
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel('Frequency [Hz]')
        ax2.set_ylim(0.5, 40)

        # Plot brainwave
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(diction['delta'], label="delta")
        ax3.plot(diction['theta'], label="theta")
        ax3.plot(diction['alpha'], label="alpha")
        ax3.plot(diction['beta'], label="beta")
        ax3.set_title('Frequency Bands')
        ax3.set_xlabel('Time [sec]')
        ax3.set_ylabel('Power')
        ax3.set_ylim(0, 400)
        ax3.legend()

        # Hiển thị hình ảnh
        plt.tight_layout()
        plt.savefig("test.png")
        plt.close()

    return diction


def STFT_transform(data):
    _, _, Zxx = sp.signal.stft(data, 512, nperseg=15 * 512, noverlap=14 * 512)
    stft_matrix = np.abs(Zxx)
    # print(stft_matrix.shape)
    return stft_matrix

def create_stft_matrix(data):
    data_segments = split_window(data)
    stft_matrices = []
    # if len(data) == 15*512:
    #    stft_matrices = STFT_transform(segment)
    for segment in data_segments:
        stft_matrices.append(STFT_transform(segment))
    stft_matrices = np.array((stft_matrices))
    return stft_matrices



def normalize_image(image):
    """
    Chuẩn hóa ảnh về khoảng [0, 1]
    """
    min_val = np.min(image)
    max_val = np.max(image)
    
    if min_val == max_val:
        return np.zeros_like(image)
    else:
        return (image - min_val) / (max_val - min_val)

glcm_feature = []
def compute_glcm(image_matrix):
    image_matrix = normalize_image(image_matrix)

    gray_image = img_as_ubyte(image_matrix)

 

    # Tính GLCM cho ảnh xám
    distances = [1]  # Các khoảng cách giữa các pixel
    angles = [0, 40*np.pi/180, 95*np.pi/180, 135*np.pi/180]  # Các góc quay
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    return glcm


def extract_features(glcm):
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = np.hstack([graycoprops(glcm, prop).ravel() for prop in properties])
    return features

def create_feature_matrix(images):
    features = []
    for image in images:
        glcm = compute_glcm(image)
        image_features = extract_features(glcm)
        features.append(image_features)
    return np.array(features)


