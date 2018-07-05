# PyMC
Monte Carlo Markov Chain sampler for a given function

Edit `testfunc.py` to implement your own model.

## Purpose
To generate most likely values for cosmological parameters  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\Omega_bh^2,&space;\Omega_mh^2,&space;\Omega_{\Lambda},&space;n_s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\Omega_bh^2,&space;\Omega_mh^2,&space;\Omega_{\Lambda},&space;n_s" title="\large \Omega_bh^2, \Omega_mh^2, \Omega_{\Lambda}, n_s" /></a>  by running MCMC with physical model and Planck data.

Current physical model calculates the theoretical peak spacing of the CMB angular Power Spectrum using theory in [Hu, Sugiyama 1994](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.51.2599) and [M. Doran, M.Lilley et. al. 2001](http://iopscience.iop.org/article/10.1086/322253/meta) and fitting formulas given in [M. Doran and M. Lilley, 2002](https://academic.oup.com/mnras/article/330/4/965/1012110).
This is comapred with the peak spacing data in [Planck satellite 2015 release](https://www.aanda.org/articles/aa/pdf/2016/10/aa26926-15.pdf).
