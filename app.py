from flask import Flask, request, jsonify
import os
import pandas as pd
from flask_cors import CORS
import json
import sys
import numpy as np
import scipy
import scipy.signal
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = Flask(__name__)

# Enable CORS globally
CORS(app)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

max_category_global = 0

def matrix_from_csv_file(file_path):
	csv_data = np.genfromtxt(file_path, delimiter = ',')
	full_matrix = csv_data[1:]
	
	return full_matrix


def get_time_slice(full_matrix, start = 0., period = 1.):
	rstart  = full_matrix[0, 0] + start
	index_0 = np.max(np.where(full_matrix[:, 0] <= rstart))
	index_1 = np.max(np.where(full_matrix[:, 0] <= rstart + period))
	
	duration = full_matrix[index_1, 0] - full_matrix[index_0, 0]
	return full_matrix[index_0:index_1, :], duration


def feature_mean(matrix):
	ret = np.mean(matrix, axis = 0).flatten()
	names = ['mean_' + str(i) for i in range(matrix.shape[1])]
	return ret, names



def feature_mean_d(h1, h2):
	ret = (feature_mean(h2)[0] - feature_mean(h1)[0]).flatten()
	names = ['mean_d_h2h1_' + str(i) for i in range(h1.shape[1])]
	return ret, names



def feature_mean_q(q1, q2, q3, q4):
	v1 = feature_mean(q1)[0]
	v2 = feature_mean(q2)[0]
	v3 = feature_mean(q3)[0]
	v4 = feature_mean(q4)[0]
	ret = np.hstack([v1, v2, v3, v4, 
				     v1 - v2, v1 - v3, v1 - v4, 
					 v2 - v3, v2 - v4, v3 - v4]).flatten()
	
	
	# Fixed naming [fcampelo]
	names = []
	for i in range(4): # for all quarter-windows
		names.extend(['mean_q' + str(i + 1) + "_" + str(j) for j in range(len(v1))])
	
	for i in range(3): # for quarter-windows 1-3
		for j in range((i + 1), 4): # and quarter-windows (i+1)-4
			names.extend(['mean_d_q' + str(i + 1) + 'q' + str(j + 1) + "_" + str(k) for k in range(len(v1))])
			 
	return ret, names




def feature_stddev(matrix):
	ret = np.std(matrix, axis = 0, ddof = 1).flatten()
	names = ['std_' + str(i) for i in range(matrix.shape[1])]
	
	return ret, names



def feature_stddev_d(h1, h2):
	ret = (feature_stddev(h2)[0] - feature_stddev(h1)[0]).flatten()
	
	# Fixed naming [fcampelo]
	names = ['std_d_h2h1_' + str(i) for i in range(h1.shape[1])]
	
	return ret, names




def feature_moments(matrix):
	skw = scipy.stats.skew(matrix, axis = 0, bias = False)
	krt = scipy.stats.kurtosis(matrix, axis = 0, bias = False)
	ret  = np.append(skw, krt)
		
	names = ['skew_' + str(i) for i in range(matrix.shape[1])]
	names.extend(['kurt_' + str(i) for i in range(matrix.shape[1])])
	return ret, names




def feature_max(matrix):
	ret = np.max(matrix, axis = 0).flatten()
	names = ['max_' + str(i) for i in range(matrix.shape[1])]
	return ret, names



def feature_max_d(h1, h2):
	ret = (feature_max(h2)[0] - feature_max(h1)[0]).flatten()
	
	# Fixed naming [fcampelo]
	names = ['max_d_h2h1_' + str(i) for i in range(h1.shape[1])]
	return ret, names


def feature_max_q(q1, q2, q3, q4):
	v1 = feature_max(q1)[0]
	v2 = feature_max(q2)[0]
	v3 = feature_max(q3)[0]
	v4 = feature_max(q4)[0]
	ret = np.hstack([v1, v2, v3, v4, 
				     v1 - v2, v1 - v3, v1 - v4, 
					 v2 - v3, v2 - v4, v3 - v4]).flatten()
	
	
	# Fixed naming [fcampelo]
	names = []
	for i in range(4): # for all quarter-windows
		names.extend(['max_q' + str(i + 1) + "_" + str(j) for j in range(len(v1))])
	
	for i in range(3): # for quarter-windows 1-3
		for j in range((i + 1), 4): # and quarter-windows (i+1)-4
			names.extend(['max_d_q' + str(i + 1) + 'q' + str(j + 1) + "_" + str(k) for k in range(len(v1))])
			 
	return ret, names


def feature_min(matrix):
	ret = np.min(matrix, axis = 0).flatten()
	names = ['min_' + str(i) for i in range(matrix.shape[1])]
	return ret, names



def feature_min_d(h1, h2):
	ret = (feature_min(h2)[0] - feature_min(h1)[0]).flatten()
	
	# Fixed naming [fcampelo]
	names = ['min_d_h2h1_' + str(i) for i in range(h1.shape[1])]
	return ret, names


def feature_min_q(q1, q2, q3, q4):

	v1 = feature_min(q1)[0]
	v2 = feature_min(q2)[0]
	v3 = feature_min(q3)[0]
	v4 = feature_min(q4)[0]
	ret = np.hstack([v1, v2, v3, v4, 
				     v1 - v2, v1 - v3, v1 - v4, 
					 v2 - v3, v2 - v4, v3 - v4]).flatten()
	
	
	# Fixed naming [fcampelo]
	names = []
	for i in range(4): # for all quarter-windows
		names.extend(['min_q' + str(i + 1) + "_" + str(j) for j in range(len(v1))])
	
	for i in range(3): # for quarter-windows 1-3
		for j in range((i + 1), 4): # and quarter-windows (i+1)-4
			names.extend(['min_d_q' + str(i + 1) + 'q' + str(j + 1) + "_" + str(k) for k in range(len(v1))])
			 
	return ret, names


def feature_covariance_matrix(matrix):

	covM = np.cov(matrix.T)
	indx = np.triu_indices(covM.shape[0])
	ret  = covM[indx]
	
	names = []
	for i in np.arange(0, covM.shape[1]):
		for j in np.arange(i, covM.shape[1]):
			names.extend(['covM_' + str(i) + '_' + str(j)])
	
	return ret, names, covM


def feature_eigenvalues(covM):	
	ret   = np.linalg.eigvals(covM).flatten()
	names = ['eigenval_' + str(i) for i in range(covM.shape[0])]
	return ret, names


def feature_logcov(covM):

	log_cov = scipy.linalg.logm(covM)
	indx = np.triu_indices(log_cov.shape[0])
	ret  = np.abs(log_cov[indx])
	
	names = []
	for i in np.arange(0, log_cov.shape[1]):
		for j in np.arange(i, log_cov.shape[1]):
			names.extend(['logcovM_' + str(i) + '_' + str(j)])
	
	return ret, names, log_cov



def feature_fft(matrix, period = 1., mains_f = 50., 
				filter_mains = True, filter_DC = True,
				normalise_signals = True,
				ntop = 10, get_power_spectrum = True):
	N   = matrix.shape[0] # number of samples
	T = period / N        # Sampling period
	
	if normalise_signals:
		matrix = -1 + 2 * (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

	fft_values = np.abs(scipy.fft.fft(matrix, axis = 0))[0:N//2] * 2 / N
	
	freqs = np.linspace(0.0, 1.0 / (2.0 * T), N//2)
	
	if filter_DC:
		fft_values = fft_values[1:]
		freqs = freqs[1:]
		
	if filter_mains:
		indx = np.where(np.abs(freqs - mains_f) <= 1)
		fft_values = np.delete(fft_values, indx, axis = 0)
		freqs = np.delete(freqs, indx)
	
	indx = np.argsort(fft_values, axis = 0)[::-1]
	indx = indx[:ntop]
	
	ret = freqs[indx].flatten(order = 'F')
	
	names = []
	for i in np.arange(fft_values.shape[1]):
		names.extend(['topFreq_' + str(j) + "_" + str(i) for j in np.arange(1,11)])
	
	if (get_power_spectrum):
		ret = np.hstack([ret, fft_values.flatten(order = 'F')])
		
		for i in np.arange(fft_values.shape[1]):
			names.extend(['freq_' + "{:03d}".format(int(j)) + "_" + str(i) for j in 10 * np.round(freqs, 1)])
	
	return ret, names


def calc_feature_vector(matrix, state):
	h1, h2 = np.split(matrix, [ int(matrix.shape[0] / 2) ])
	q1, q2, q3, q4 = np.split(matrix, 
						      [int(0.25 * matrix.shape[0]), 
							   int(0.50 * matrix.shape[0]), 
							   int(0.75 * matrix.shape[0])])

	var_names = []	
	
	x, v = feature_mean(matrix)
	var_names += v
	var_values = x
	
	x, v = feature_mean_d(h1, h2)
	var_names += v
	var_values = np.hstack([var_values, x])

	x, v = feature_mean_q(q1, q2, q3, q4)
	var_names += v
	var_values = np.hstack([var_values, x])
	
	x, v = feature_stddev(matrix)
	var_names += v
	var_values = np.hstack([var_values, x])
	
	x, v = feature_stddev_d(h1, h2)
	var_names += v
	var_values = np.hstack([var_values, x])
	
	x, v = feature_moments(matrix)
	var_names += v
	var_values = np.hstack([var_values, x])
	
	x, v = feature_max(matrix)
	var_names += v
	var_values = np.hstack([var_values, x])
	
	x, v = feature_max_d(h1, h2)
	var_names += v
	var_values = np.hstack([var_values, x])

	x, v = feature_max_q(q1, q2, q3, q4)
	var_names += v
	var_values = np.hstack([var_values, x])
	
	x, v = feature_min(matrix)
	var_names += v
	var_values = np.hstack([var_values, x])
	
	x, v = feature_min_d(h1, h2)
	var_names += v
	var_values = np.hstack([var_values, x])

	x, v = feature_min_q(q1, q2, q3, q4)
	var_names += v
	var_values = np.hstack([var_values, x])
	
	x, v, covM = feature_covariance_matrix(matrix)
	var_names += v
	var_values = np.hstack([var_values, x])
	
	x, v = feature_eigenvalues(covM)
	var_names += v
	var_values = np.hstack([var_values, x])
	
	x, v, log_cov = feature_logcov(covM)
	var_names += v
	var_values = np.hstack([var_values, x])
	
	x, v = feature_fft(matrix)
	var_names += v
	var_values = np.hstack([var_values, x])
	
	if state != None:
		var_values = np.hstack([var_values, np.array([state])])
		var_names += ['Label']

	return var_values, var_names

def generate_feature_vectors_from_samples(file_path, nsamples, period, 
										  state = None, 
										  remove_redundant = True,
										  cols_to_ignore = None):

	matrix = matrix_from_csv_file(file_path)
	
	t = 0.
	previous_vector = None
	ret = None
	
	while True:
		try:
			s, dur = get_time_slice(matrix, start = t, period = period)
			if cols_to_ignore is not None:
				s = np.delete(s, cols_to_ignore, axis = 1)
		except IndexError:
			break
		if len(s) == 0:
			break
		if dur < 0.9 * period:
			break
		
		ry, rx = scipy.signal.resample(s[:, 1:], num = nsamples, 
								 t = s[:, 0], axis = 0)
		
		t += 0.5 * period
		
		r, headers = calc_feature_vector(ry, state)
		
		if previous_vector is not None:
			feature_vector = np.hstack([previous_vector, r])
			
			if ret is None:
				ret = feature_vector
			else:
				ret = np.vstack([ret, feature_vector])
				
		previous_vector = r
		if state is not None:
			previous_vector = previous_vector[:-1] 

	feat_names = ["lag1_" + s for s in headers[:-1]] + headers
	
	if remove_redundant:
		to_rm = ["lag1_mean_q3_", "lag1_mean_q4_", "lag1_mean_d_q3q4_",
		         "lag1_max_q3_", "lag1_max_q4_", "lag1_max_d_q3q4_",
				 "lag1_min_q3_", "lag1_min_q4_", "lag1_min_d_q3q4_"]
		
		for i in range(len(to_rm)):
			for j in range(ry.shape[1]):
				rm_str = to_rm[i] + str(j)
				idx = feat_names.index(rm_str)
				feat_names.pop(idx)
				ret = np.delete(ret, idx, axis = 1)
	return ret, feat_names


def gen_training_matrix(full_file_path, cols_to_ignore):
	
    vectors, header = generate_feature_vectors_from_samples(file_path = full_file_path, 
														        nsamples = 150, 
																period = 1.,
														        remove_redundant = True,
																cols_to_ignore = cols_to_ignore)

    FINAL_MATRIX = vectors	
    df = pd.DataFrame(FINAL_MATRIX)

    # print(header)
    print(df.shape)
    # print(df.head())
    # print(df[0])
	
    return df

def predictionss(pickle_file_path, pca_file_path, scaler_file_path, test_dataframe):
	
    model = pickle.load(open(pickle_file_path, 'rb'))
	
    pca_loaded = pickle.load(open(pca_file_path, 'rb'))
	
    scaler_loaded = pickle.load(open(scaler_file_path, 'rb'))
	
    scaled_new_data = scaler_loaded.transform(test_dataframe)
    pca_new_data = pca_loaded.transform(scaled_new_data)
	
    print(pca_new_data)
    
    predictions = np.argmax(model.predict(pca_new_data), axis=1)
	
    data = np.array(predictions)

    total_count = len(data)
    counts = {value: np.sum(data == value) for value in np.unique(data)}

    percentages = {key: (value / total_count) * 100 for key, value in counts.items()}
	
    max_category = max(percentages, key=percentages.get)
	
    print(max_category)

    results = []
    for category, percentage in percentages.items():
        state = "Concentrated" if category == 2 else "Relaxed" if category == 0 else "Neutral"
        results.append({"state": state, "percentage": f"{percentage:.2f}%"})

    return {"results": results, "max_result": {"state": str(max_category), "percentage": f"{percentages[max_category]:.2f}%"}}
	
TEMP_STORAGE_FILE = "temp_results.json"

def save_results_to_temp(results):
    """
    Save the full response to a temporary file.
    """
    with open(TEMP_STORAGE_FILE, "w") as f:
        json.dump(results, f)

def load_results_from_temp():
    """
    Load the full response from the temporary file.
    """
    if os.path.exists(TEMP_STORAGE_FILE):
        with open(TEMP_STORAGE_FILE, "r") as f:
            data = json.load(f)
            return data
    return None 

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed.'}), 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
	
        full_file_path = filepath
        dataframe = gen_training_matrix(full_file_path, cols_to_ignore = -1)
        
        pickle_file_path = "final_model.pkl"
        pca_file_path = "pca_custom.pkl"
        scaler_file_path = "scaler_custom.pkl"
        
        results = predictionss(pickle_file_path, pca_file_path, scaler_file_path, dataframe)
		
        response = {
            "results": results["results"],
            "highlight": results["max_result"]["state"]
        }

        # Save the full response to temporary storage
        save_results_to_temp(response)

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/get-highlight', methods=['GET'])
def get_highlight():
    saved_results = load_results_from_temp()
    if saved_results is None:
        return jsonify({"error": "No data available. Upload a file first."}), 400

    return jsonify({"value" : int(saved_results["highlight"])})

if __name__ == '__main__':
    app.run(debug=True)
