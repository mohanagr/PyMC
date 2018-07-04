import numpy as np
import configparser

# Add a separate function to read ini file if it gets too complicated

def main():

	config = configparser.ConfigParser()
	config.read('./config.ini')

	initparams = config['INIT']
	steps = initparams.getint('steps')
	outfile = initparams['outfile']
	infile = initparams['infile']
	print(steps, outfile, infile)

	n_params = config['PARAMS'].getint('number')

	ranges = np.zeros((n_params, 2))

	# Read input file containing parameter bounds
	with open(infile, 'r') as f:
		for i in range(0, n_params):
			l = f.readline()
			low, high = [float(val) for val in l.split()]

	chain1 = mcmc(steps, y_data, y_err)
