"""
Generic auxiliary code for assistance with conducting experiments.
"""
# 
# Copyright (c) 2017 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the MIT License. See the LICENSE file for details.


import pickle #, bz2
from time import sleep

import numpy as np



# ==================================== ## ==================================== #

def initialize_rng(seed=None, seeder_rng=None, *args, **kwargs):
	"""
	Initializes a separate numpy random number generator object. Its state
	is initialized using `seed`, if provided, or a newly generated seed if not.
	
	Should a random number generator function be provided (`seeder_rng`), it
	will be used to generate the seed. Otherwise, `numpy.random.randint` will be
	used as the seeder. Valid functions to provide in `seeder_rng` include
	`random.randrange` and `numpy.random.randint`, or equivalent functions from
	independent generators, given by `random.Random` or `np.random.RandomState`.
	"""
	if seed is None:
		random_integer = np.random.randint if seeder_rng is None else seeder_rng
		# np.random.randint converts its arguments internally to C signed longs
		# (32 bits, ranging in [-2**31, 2**31-1])
		lim = 2**31
		seed = random_integer(-lim, lim-1)
		# RandomState requires a seed in [0, 2**32-1]
		seed += lim
	
	rng = np.random.RandomState(seed)
	return rng, seed



# ==================================== ## ==================================== #

class online_variance(object):
	"""
	Welford's algorithm to compute variance (and mean) incrementally.
	
	Adapted from:
	* http://stackoverflow.com/a/5544108
	  (shared by unutbu; user contributions licensed under CC BY-SA 3.0)
	See also:
	* https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
	* http://stackoverflow.com/q/32135572
	* http://stackoverflow.com/a/1348615
	"""
	def __init__(self, iterable=None, ddof=1):
		# ddof=0: population variance, ddof=1: sample variance
		# NOTE: default setting in numpy.var() is ddof=0
		# https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html
		self.ddof = ddof
		self.n, self.mean, self.M2 = 0, 0.0, 0.0
		if iterable is not None:
			for datum in iterable:
				self.include(datum)
	
	def include(self, datum):
		self.n += 1
		self.delta = datum - self.mean
		self.mean += self.delta / self.n
		self.M2 += self.delta * (datum - self.mean)
		self.variance = self.M2 / (self.n - self.ddof)
	
	@property
	def std(self):
		return np.sqrt(self.variance)



# ==================================== ## ==================================== #

def safe_dump(data, fname, append=False):
	"""
	Pickle `data` into the file of name `fname` (possibly appending into it).
	
	Should another process be concurrently accessing the file's data (such as
	through `pickle_iterator`/`batch_iterator`), the function sleeps and
	retries, until the writing succeeds.
	"""
	mode = 'ab' if append else 'wb'
	
	while True:
		try:
#			with bz2.BZ2File(fname, mode) as f:
			with open(fname, mode) as f:
				pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
			return
		except IOError:
			# some other process may be accessing the file.. retry
			sleep(10)
	# https://docs.python.org/3/library/functions.html#open
	# https://docs.python.org/3/library/bz2.html#bz2.BZ2File
	# https://docs.python.org/3/library/pickle.html



def pickle_iterator(fname):
	"""
	Iterate over the contents of a pickle built by incrementally appending data.
	See: safe_dump(..., append=True).
	"""
#	with bz2.BZ2File(fname, 'rb') as f:
	with open(fname, 'rb') as f:
		while True:
			try:
				yield pickle.load(f)
			except EOFError:
				break


def batch_iterator(fname, stop_at=None, verbose=True):
	"""
	Iterates over all items of a sequence stored in batches in a pickle,
	or alternatively only up to the n-th element.
	
	Examples
	--------
	>>> safe_dump([1, 2], 'TEST', append=True)
	>>> safe_dump([3, 4], 'TEST', append=True)
	>>> for i in batch_iterator('TEST', verbose=False): print(i, end=' ')
	1 2 3 4
	"""
	j = 0
	for i,batch in enumerate(pickle_iterator(fname)):
		if verbose:
			# write status message indicating a batch was read from `fname`
			msg = str(i+1) if (i+1)%10==0 else (':' if (i+1)%5==0 else '.')
			print(msg, end='')
		for data in batch:
			yield data
			j += 1
			if j == stop_at:
				raise StopIteration
	if verbose:
		print() # newline


