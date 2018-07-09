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
		self._notgauss = list()

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
		print("PARAM RANGES\n", self.param_ranges)

	def _propose(self):

		means = self.params.copy()
		var = self.variance

		for i in self.gaussian.keys():
			means[i] = float(self.gaussian[i][0])
			var[i] = float(self.gaussian[i][1]) 

		# print("MEANS", means)
		# print("VARIANCES", var)
		flag = 1
		step = np.zeros(4)
		
		while(True):
			flag = 0
			for i in range(0, self.n_params):
				step[i] = np.random.normal(means[i], var[i])
				low = self.param_ranges[i, 0]
				high = self.param_ranges[i, 1]
				if(((low-step[i])>1.0E-8)  or ((step[i]-high) > 1.0E-08)):
					flag = 1
			
			if(flag==0):
				break

		return step

	def _init_params(self):

		self._read()

		# Set initial variance for proposal steps
		self.variance = (2.3/(float(self.n_params))**0.5)*np.ones(self.n_params)

		self.params = np.zeros(self.n_params)

		for i, param in enumerate(self.params):

			if(i in self.gaussian.keys()):
				self.params[i] = np.random.normal(self.gaussian[i][0], self.gaussian[i][1])
			else:
				self._notgauss.append(i)
				self.params[i] = np.random.uniform(low = self.param_ranges[i, 0], high = self.param_ranges[i, 1])

		print("After initialization", self.params)

	def _adjust_sigma(self):

		ratio = self.rej/(self.rej + self.acc)

		# Too many rejections. (reduce step size, be less risky)
		if(ratio > 0.8):

			for i in range(0, self.n_params):
				if (i in self._notgauss):
					self.variance[i] = self.variance[i]*0.8
		
		# Too many accepted. (Inc step size, be more risky)
		elif(ratio < 0.4):

			for i in range(0, self.n_params):
				if (i in self._notgauss):
					self.variance[i] = self.variance[i]*1.2

		# print(self.variance)
			
	def run_chain(self, fileobj):

		for i in range(0, self.max_steps):

			# print("PARAMS AT BEGINNING OF RUN CHAIN", self.params)
			
			# Calculate initial CHISQUARE statistic
			y_calc1 = self.func(self.params)
			self.chi2 = chisquare.chi2(self.data, y_calc1, self.data_err)

			# print("from run chain", self.proposal)
			step = self._propose()
			y_calc2 = self.func(step)

			chi2_new = chisquare.chi2(self.data, y_calc2, self.data_err)

			cmp = (self.chi2 - chi2_new)

			if( cmp > 1.0E-08):
				self.acc += 1
				self.params = step
				self.chi2 = chi2_new
				self.weight = 0
				# print("AC NEW PARAM", self.params)
				retstr = ["ACC", 1, self.chi2]
			elif(-cmp > 1.0E-08):
				toss = np.random.uniform()
				alpha = np.exp((-chi2_new + self.chi2)/2.0)

				# This step will be very less probable if CHISQ values are too large.
				# i.e. hardly any chance to accept larger (worse likelihood) chi, esp. when the chi is in 1000s
				# Follows common sense

				if( (alpha - toss) > 1.0E-08):
					self.acc += 1
					self.params = step
					self.chi2 = chi2_new
					self.weight = 0
					retstr = ["TOS", 1, self.chi2]
					# print("AC NEW PARAM from TOSS")
				else:
					# print("FROM REJECT", self.params, step)
					self.weight += 1
					self.rej += 1
					retstr = ["REJ", self.weight, self.chi2]

			line = "{} {:3d} {:10.7f}".format(*retstr) + self.n_params*"{:10.7f}".format(*self.params) + "\n"

			fileobj.write(line)

			if(i%100 == 0):
				self._adjust_sigma()

		print("Total accepted : ", self.acc)
		print("Total rejected : ", self.rej)


	def generate(self, filename = None):

		if(not os.path.exists("./Output/")):
			os.makedirs("./Output/")

		if(filename is not None):
			self.outfile = filename

		with open("./Output/"+self.outfile, 'w') as f:
			self.run_chain(f)