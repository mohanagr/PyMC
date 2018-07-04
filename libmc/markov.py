import numpy as np
import chisquare
class mcmc():

	def __init__(self, steps, y_data, y_err, target_func, outfile, n_params, params_range):

		self.max_steps = steps
		self.n_params = n_params
		self.ranges = params_range
		self.data = y_data
		self.data_err = y_err
		self.outfile = outfile

		# Set initial parameters
		self._init_params()

		
		# Set initial variance for proposal steps
		self.variance = (2.3/(float(n_params))**0.5)*np.ones(n_params)

		# Set up counters
		self.acc = 0
		self.rej = 0	
		self.weight = 0

	def _propose():

		means = np.zeros(self.n_params)
		var = np.diag(self.variance)

		# ADD PRIORS HERE

		step = np.random.multivariate_normal(means, var , 1)

		self.propsal = self.params + step

	def _init_params():

			self.params = np.ones(self.n_params)

			for i, param in enumerate(self.params):

				self.params[i] = np.random.uniform(low = self.params_range[i,0], high = self.params_range[i, 1])

			# PRIOR STEP (Replace uniform by Gaussian for some variables)

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




