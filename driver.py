import numpy as np
import configparser

# Add a separate function to read ini file if it gets too complicated

def get_input(datafile)

	# Data reading

	with open(datafile) as f:
	lines = f.readlines()

	size = len(lines)
	lval = np.zeros((size, 2))

	for index, line in enumerate(lines):
		l, err = [float(num) for num in line.split()]
		lval[index, :] = [l, err]

	planck_peaks = [lval[2*m-2, 0] for m in range(1,9)]
	planck_peaks_err = [lval[2*m-2, 1] for m in range(1,9)]

	#----------------------------------------------------------------
	# Simulation only for first 3 peaks (M. Doran, M. Lilley, 2002) |
	#----------------------------------------------------------------
	y_data = planck_peaks[0:3]
	y_data_err = planck_peaks_err[0:3]

	return y_data, y_data_err

def main():

	config = configparser.ConfigParser()
	config.read('./config.ini')

	initparams = config['INIT']
	steps = initparams.getint('steps')
	outfile = initparams['outfile']
	infile = initparams['infile']
	datafile = initparams['datafile']
	print(steps, outfile, infile)

	n_params = config['PARAMS'].getint('number')

	ranges = np.zeros((n_params, 2))

	# Read input file containing parameter bounds
	with open(infile, 'r') as f:
		lines = f.redlines()

	for i, line in enumerate(lines):
		ranges[i,:] = [float(val) for val in line.split()]

	# Get experimental data and error
	y_data, y_err = get_input()

	# Initiate chain
	chain1 = mcmc(steps, y_data, y_err)
