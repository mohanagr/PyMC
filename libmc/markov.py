import numpy as np
import os
from . import chisquare

class mcmc():

	def __init__(self, steps, y_data, y_err, target_func, infile, outfile):

		self.max_steps = steps
		self.data = y_data
		self.data_err = y_err
		self.infile = infile
		self.outfile = outfile

		self.func = target_func

		# Keeping track of parameters with priors
		self.gaussian = dict()
		self.uniform = list()

		# Set initial parameters
		self._init_params()

		# Set up counters
		self.acc = 0
		self.rej = 0	
		self.weight = 0

	def _read(self):

		# Read input file containing parameter bounds
		with open(self.infile, 'r') as f:
			lines = f.readlines()

		self.n_params = len(lines)

		print("No. of params are ", self.n_params)

		self.param_ranges = np.zeros((self.n_params, 2))

		for i, line in enumerate(lines):
			values = [float(val) for val in line.split()]
			self.param_ranges[i,:] = values[0:2]
			flag = int(values[2])

			if(flag == 1):
				self.gaussian[i] = values[3:5]
			elif(flag == 2):
				self.uniform.append(i)
		print("SELF GAUSSIAN", self.gaussian)
		print("PARAM RANGES", self.param_ranges)

	def _propose(self):

		means = self.params
		var = self.variance

		for i in self.gaussian.keys():
			means[i] = self.gaussian[i][0]
			var[i] = self.gaussian[i][1] 

		print("MEANS", means)
		print("VARIANCES", var)
		flag = 1
		step = np.zeros(4)
		# while(flag>0):
			# flag = 0
			# print("in while loop", step)
		for i in range(0, self.n_params):
			step[i] = np.random.normal(means[i], var[i])
			low = self.param_ranges[i, 0]
			high = self.param_ranges[i, 1]
				# print("low", low, "high", high)
				# if((low-step[i]>1.0E-8)  or (step[i]-high > 1.0E-08)):
					# flag = 1
			# print(flag)

		self.proposal = step

	def _init_params(self):

		self._read()

		# Set initial variance for proposal steps
		self.variance = 0.01*(2.3/(float(self.n_params))**0.5)*np.ones(self.n_params)

		self.params = np.ones(self.n_params)

		for i, param in enumerate(self.params):

			if(i in self.gaussian.keys()):
				self.params[i] = np.random.normal(self.gaussian[i][0], self.gaussian[i][1])
			else:
				self.params[i] = np.random.uniform(low = self.param_ranges[i, 0], high = self.param_ranges[i, 1])

		print("After init", self.params)

			

	def _run_chain(self):

		print("PARAMS AT BEGINNING OF RUN CHAIN", self.params)
		# Calculate initial CHISQUARE statistic
		y_calc1 = self.func(self.params)
		print(y_calc1)
		self.chi2 = chisquare.chi2(self.data, y_calc1, self.data_err)
		print("chi2 originnal: ", self.chi2)

		self._propose()	

		# print("from run chain", self.proposal)
		y_calc2 = self.func(self.proposal)

		chi2_new = chisquare.chi2(self.data, y_calc2, self.data_err)
		print("chi2 new: ", chi2_new)
		if(chi2_new < self.chi2):
			self.acc += 1
			retstr = [1, chi2_new/2.0, self.proposal]
			self.params = self.proposal
			self.chi2 = chi2_new
			self.weight = 0
			print("AC NEW PARAM", self.params)
		else:
			toss = np.random.uniform()
			alpha = np.exp((-chi2_new + self.chi2)/2.0)

			if(toss < alpha):
				self.acc += 1
				retstr = [1, chi2_new/2.0, self.proposal]
				self.params = self.proposal
				self.chi2 = chi2_new
				self.weight = 0
				print("AC NEW PARAM", self.params)
			else:
				self.weight += 1
				self.params = self.params
				retstr = [self.weight, self.chi2/2.0, self.params]
				self.rej += 1
				print("REJ")
					
		print("RETSTR", retstr)
		return retstr

	def generate(self, filename = None):

		if(not os.path.exists("./Output/")):
			os.makedirs("./Output/")

		if(filename is not None):
			self.outfile = filename

		with open("./Output/"+self.outfile, 'w') as f:

			for i in range(0, self.max_steps):

				label, likelihood, params = self._run_chain()
				print("LABEL IS", label)
				filestr = '{0:3.1f} {1:10.7f}'.format(float(label), likelihood)
				for param in params:
					filestr = filestr + '{:10.7f}'.format(param)
				# print("FILESTR IS", filestr)
				f.write(filestr+'\n')




