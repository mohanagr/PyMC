import numpy as np

def chi2(y_data, y_func, sigma_y):
	"""
		Returns CHI SQUARE statistic

		Parameters
		----------
		arg1 : numpy array
		(Experimental) Data to be tested against
		arg2 : numpy array
		Values from function approximating the data
		arg3 : numpy array
			1 SIGMA certainity in each of the measured values

		Returns
		-------
		numpy scalar
		Calculated CHI SQUARE
	"""
	arr = ((y_data - y_func)/sigma_y)**2
	chi2 = np.sum(arr)

	return chi2