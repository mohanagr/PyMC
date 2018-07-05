import numpy as np
from scipy.special import integrate

def func(params):
	ohmbh2, ohmm, h, n = params

	#----------------------------------------------------------
	# comoving distance to surface of last scattering estimate|
	#----------------------------------------------------------
	# Matter Dominated era assumed at LSS                     |
	#----------------------------------------------------------

	ohmLambda = 1 - ohmm
	ohmmh2 = ohmm * h * h
	#ohmr = 9.89E-5
	
	H0 = h*100 / 2.99792458E05

	#print(ohmm)

	def dist(z):

		r = 1/(ohmLambda + ohmm * (1 + z)**3 )**0.5
		
		return r

	d_ls = integrate.quad(dist, 0.0, 1100.0)

	rs = get_sound_horizon(ohmbh2,ohmmh2, 1100)

	#---------------------------------------------------------
	# Calculate l_a                                          |
	#---------------------------------------------------------
	la = np.pi * d_ls[0] / (rs * H0)

	phi, del2, del3 = getShiftParams(0, ohmbh2, n)

	y_th = np.zeros(3)

	y_th[0] = la*(1 - phi)
	y_th[1] = la*(2 - phi - del2)
	y_th[2] = la*(3 - phi - del3)

	return y_th

def getShiftParams(ohm_ls, ohmbh2, n):

	a1 = 0.286 + 0.626*ohmbh2
	a2 = 0.1786 - 6.308*ohmbh2 + 174.9*ohmbh2**2 - 1168*ohmbh2**3
	phi_ = (1.466 - 0.466*n)*(a1*r_**a2 + 0.291*ohm_ls)

	c0 = -0.1 + (0.213 - 0.123*ohm_ls) * np.exp(ohm_ls*(63.6 - 52) - ohmbh2)
	c1 = 0.063*np.exp(-3500*ohmbh2**2) + 0.015
	c2 = 6.0E-6 + 0.137*(ohmbh2**2 - 0.07)**2
	c3 = 0.8 + 2.3*ohm_ls + (70 -126*ohm_ls)*ohmbh2
	del2 = c0 - c1*r_ - c2*r_**(-c3) + 0.05*(n - 1)


	d1 = 9.97 + (3.3 - 3*ohm_ls)*ohmbh2
	d2 = 0.0016 - 0.0067*ohm_ls + (0.196 - 0.22*ohm_ls)*ohmbh2 + (2.25 + 2.77*ohm_ls)/(100000*ohmbh2)
	del3 = 10 - d1*r_**d2 + 0.08*(n - 1)
	
	return phi_, del2, del3

def get_sound_horizon(ohmbh2,ohmmh2, zrec):
	
	r = 0.0459 * (1 + zrec) / (ohmmh2 * 1100)
	R = 27.6 * (ohmbh2 * 1100) / (1 + zrec)
	
	k = 2998 * 2 / ( (1 + zrec)*(3 * ohmmh2 * R) )**0.5
	
	w = np.log(( (1 + R)**0.5 + (R + r*R)**0.5 ) / ( 1 + (r*R)**0.5 ))
	
	rs = k * w
	
	return rs