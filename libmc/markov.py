import numpy as np
import chisquare
class mcmc():

	def __init__(self, steps, y_data, y_err, target_func, infile, outfile):

		self.max_steps = steps
		self.n_params = n_params
		self.ranges = params_range
		self.data = y_data
		self.data_err = y_err
		self.infile = infile
		self.outfile = outfile

		# Keeping track of parameters with priors
		self.gaussian = list()
		self.uniform = list()

		# Set initial parameters
		self._init_params()

		
		# Set initial variance for proposal steps
		self.variance = (2.3/(float(n_params))**0.5)*np.ones(n_params)

		# Set up counters
		self.acc = 0
		self.rej = 0	
		self.weight = 0

	def _read():

		# Read input file containing parameter bounds
		with open(infile, 'r') as f:
			lines = f.readlines()

		self.n_params = len(lines)

		ranges = np.zeros((n_params, 2))

		for i, line in enumerate(lines):
			values = [float(val) for val in line.split()]
			ranges[i,:] = values[0:2]
			flag = int(values[2])

			if(flag == 1):
				self.gauss_params = values[3:5]
				self.gaussian.append(i)
			elif(flag == 2):
				self.uniform.append(i)

	def _propose():

		means = self.params
		var = np.diag(self.variance)

		# OMEGAbh^2 from BBN
		means[0] = 0.022
		var[0] = 0.002

		# OMEGAm from BAO
		means[1] = 0.303
		var[1] = 0.040

		step = np.random.multivariate_normal(means, var , 1)

		self.propsal = step

	def _init_params():

			self.params = np.ones(self.n_params)

			for i, param in enumerate(self.params):

				if(i in self.gaussian):
					self.params[i] = np.random.normal(self.gauss_params[0], self.gauss_params[1])

				self.params[i] = np.random.uniform(low = self.params_range[i,0], high = self.params_range[i, 1])

			# Calculate initial CHISQUARE statistic
			self.chi2 = chisquare.chi2(self.data, y_calc1, self.data_err)

	def _run_chain():

		y_calc1 = self.func(self.params)
		y_calc2 = self.func(self.propsal)

		chi2_new = chisquare.chi2(self.data, y_calc2, self.data_err)

		if(chi2_new < self.chi2):
			self.acc += 1
			retstr = [1.0, chi2_new/2.0, self.proposal]
			self.params = self.proposal
			self.chi2 = chi2_new
			self.weight = 0
		else:
			toss = np.random.uniform()
			alpha = np.exp((-chi2_new + chi2/2.0))

			if(toss < alpha):
				self.acc += 1
				retstr = [1.0, chi2_new/2.0, self.proposal]
				self.params = self.proposal
				self.chi2 = chi2_new
				self.weight = 0
			else:
				retstr = [self.weight, self.chi2/2.0, self.params]
				self.rej += 1
				self.weight += 1	

		return retstr

	def generate(self, filename = self.outfile):

		if(not os.path.exists("./Output/")):
			os.makedirs("./Output/" + filename)

		with open("./Output/"+filename, 'w') as f:

			for i in range(0, self.max_steps):

				label, likelihood, params = self._run_chain()
				filestr = '{0:10.7f}{1:10.7f}'.format(label, likelihood)
				for param in params:
					filestr = filestr + '{:10.7f}'.format(param)

				f.write(filestr)




