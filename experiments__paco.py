"""
Experimental evaluation, over the GTOC5 trajectory optimization problem, of the
algorithms:
  * P-ACO (Population-based Ant Colony Optimization)
  * Beam P-ACO
  * Stochastic Beam
  * Beam Search
"""
# 
# Copyright (c) 2017 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the MIT License. See the LICENSE file for details.


import os
from pprint import pprint

from tqdm import tqdm, trange

from paco_traj import *
from experiments import initialize_rng, safe_dump



# ==================================== ## ==================================== #
# ------------------------------------ # Config./instantiate GTOC5 path handler

def init__path_handler(multiobj_evals=True, **kwargs):
	"""
	Configure and instantiate the GTOC5 "path handler", used by the search
	methods to create, inspect and evaluate missions.
	"""
	# initial asteroid at which all trajectories/ant walks will start
	starting_ast = 1712		# 2001 GP2
#	starting_ast = 4893		# 2007 UN12
	
	
	## --- Parameters configuring the heuristic function
	
	# phasing indicator used to rate destination asteroids
#	rating = rate__orbital
	rating = rate__orbital_2
	
	# reference transfer time (in days) used in the Orbital phasing indicator
	ref_dT = 125
	#ref_dT = 250
	
	# "greediness exponent" used in the `heuristic` function
	#gamma = 25
	gamma = 50
	#gamma = 100
	
	
	## --- Parameters configuring the addition of new legs to missions
	
	# optimization by grid search, driven by min dV
	add_ast_args = dict(grid=True, obj_fun=gtoc5_rendezvous)
	# optimization by grid search, driven by softmin aggregation
#	add_ast_args = dict(grid=True, obj_fun=gtoc5_rendezvous_agg)
	
	# optimization by minimize_scalar, driven by min dV
#	add_ast_args = dict(grid=False, obj_fun=gtoc5_rendezvous)
	# optimization by minimize_scalar, driven by softmin aggregation
#	add_ast_args = dict(grid=False, obj_fun=gtoc5_rendezvous_agg)
	
	# replicating the settings used by the ACT/GOL team during GTOC5
#	add_ast_args = dict(grid=True, obj_fun=gtoc5_rendezvous)
#	add_ast_args.update(dict(leg_dT_bounds=[100., 490.], nr_evals=40))
#	add_ast_args.update(dict(leg_dT_bounds=[150., 690.], nr_evals=55))
	
	# indication for `lambert_optimize_dt` to choose transfer time by picking
	# a random solution from the Pareto front of (dT, dV) solutions
	# (requires `grid=True`)
#	add_ast_args.update(dict(grid=True, random_pareto=True))
	
	# switching off the caching of rendezvous legs
#	add_ast_args['use_cache'] = False
	
	
	## --- Instantiate and return the path handler
	
	path_args = dict(starting_ast=starting_ast,
	                 ratingf=rating, ref_dT=ref_dT, gamma=gamma,
	                 add_ast_args=add_ast_args)
	path_args.update(kwargs)
	
	# `multiobj_evals` specifies how to perform mission evaluation and sorting:
	# * if False: by score + resource availability
	# * if True: by Pareto dominance
	return (gtoc5_ant_pareto if multiobj_evals else gtoc5_ant)(**path_args)
	


# ==================================== ## ==================================== #
# ------------------------------------ # Experiment controller

class experiment(object):
	"Experiment controller"
	
	def __init__(self, path_handler, nr_runs=100, log_data_every=2, 
	             max_nr_legs=None, max_nr_gens=None, path='', extra_info=None,
	             **kwargs):
		
		self.nr_runs = nr_runs
		self.max_nr_legs = max_nr_legs
		self.max_nr_gens = max_nr_gens
		self.log_data_every = log_data_every
		
		self.set_parameters(**kwargs)
		
		# instantiate the search method
		self.aco = self.aco_class(
			nr_nodes=len(asteroids), path_handler=path_handler,
			random_state=None, **self.aco_args)
		
		if self.pareto_elite:
			assert 'pareto' in path_handler.__class__.__name__, \
				'Pareto Elitism requires multi-objective evaluations.'
		
		self.set_filename(path, extra_info)
#		pprint(self.__dict__)
#		pprint(self.aco.__dict__)
		
	
	def set_parameters(self, variant='P-ACO', pareto_elite=False, **kwargs):
		"Parameter settings for the experimental setup's different variants."
		
		self.variant = variant
		self.pareto_elite = pareto_elite
		
		self.aco_class = {
			(True, False) : paco,
			(True, True) : paco_pareto,
			(False, False) : beam_paco,
			(False, True) : beam_paco_pareto,
			}[(variant == 'P-ACO', pareto_elite)]
		
		# default parameter settings
		self.aco_args = dict(pop_size=3, ants_per_gen=25, alpha=1., beta=5.,
		                     prob_greedy=0.5, use_elitism=True)
		
		# parameter changes for the Beam Search variants
		# ('beam_width' and 'branch_factor' should be defined via `kwargs`)
		diff = {
			# Hybridization of Beam Search and P-ACO
			'Beam P-ACO' : dict(),
			
			# Beam Search variant where successor nodes are picked
			# non-deterministically from a distribution defined solely by the
			# heuristic function. Equivalent to 'Beam P-ACO', in that it's a
			# beam search performing random restarts, but here with no knowledge
			# transfer between restarts (alpha=0).
			'Stochastic Beam' : dict(alpha=0., beta=1.),
			
			# Standard (deterministic) Beam Search
			'Beam Search' : dict(alpha=0., beta=1., prob_greedy=1.0),
			
			}.get(self.variant, {})
		
		self.aco_args.update(diff)
		self.aco_args.update(kwargs)
		
		# 'beam_width' is accepted as an alias to 'ants_per_gen'.
		# to enforce consistency, ensure 'ants_per_gen' is always set.
		if 'beam_width' in self.aco_args:
			self.aco_args['ants_per_gen'] = self.aco_args['beam_width']
		
	
	def set_filename(self, path='', extra_info=None):
		"Set path and name of the file into which to save the results."
		self.path = path if path[-1] == '/' else (path + '/')
		# create directory to store results (in case it doesn't exist yet)
		if not os.path.exists(path):
			os.makedirs(path)
		
		bf = self.aco_args.get('branch_factor', None)
		if self.pareto_elite:
			pareto_elite_str = ' (pareto %df)' % self.aco.nr_elite_fronts
		self.filename = \
			'pop_size={pop_size:d}, ants_per_gen={ants_per_gen:d}, ' \
			'alpha={alpha:.1f}, beta={beta:.1f}, prob_greedy={prob_greedy:.2f}'\
			', elitism={use_elitism}{_branch_factor}, ' \
			'variant={variant:s}{pareto_elite}{extra_info}.pkl'.format(
				_branch_factor=(', branch_factor=%d' % bf) if bf else '',
				variant=self.variant,
				pareto_elite='' if not self.pareto_elite else pareto_elite_str,
				extra_info=(', %s' % extra_info) if extra_info else '',
				**self.aco_args)
		
	
	def show_setup(self):
		"Display the experimental setup's configuration."
		print('\nvariant: ' + self.variant, end='\n\n')
		print(self.path + '\n' + self.filename, end='\n\n')
		pprint(self.aco_args); print('')
		print(self.aco.path.__class__, self.aco_class, end='\n\n')
		
	
	def print_best(self):
		"Print information about the best sequence found to date in a run."
		print('')
#		(q, m) = self.aco.best
#		print(seq(m, incl_flyby=False))
#		q = (score(m), resource_rating(m), final_mass(m), tof(m) * DAY2YEAR)
#		print(quality_nt(*q))
		if self.pareto_elite:
			msg = 'Size of the Pareto Elite archive: %d' % len(self.aco.elite)
			pf = self.aco.path.sort(self.aco.elite, f=1)
			msg += ', Size of the Pareto front: %d' % len(pf)
			obj = np.array([q[1:] for (q, m) in pf])
			hv = hypervolume(obj, reference_point=(3500., 15.))
			msg += ', hypervolume: %f' % hv
			print(msg)
		st = self.aco.path.stats
		w = st.when_best_found()
		print(st.seq[w])
		print(quality_nt(*st.quality[w]))
		print(when_nt(*st.when[-1]))
		print('')
		
	
	def stats_best(self):
		"Obtain statistics about the best sequence found to date in a run."
		if 'pareto' not in self.aco.__class__.__name__:
			(q, m) = self.aco.best
		else:
			# show stats on the solution in the Pareto front with highest
			# resource rating
			(q, m) = max(self.aco.best, key=lambda i: resource_rating(i[1]))
		return '[Score: %2d, Rating: %.5f, Mass: %7.3f, Time: %6.3f%s]' % (
			score(m), resource_rating(m), final_mass(m), tof(m) * DAY2YEAR,
			'; |e|=%d' % len(self.aco.elite))
		
	
	def run(self, seed=None):
		"Perform a full, independent run."
		self.aco.random, seed = initialize_rng(seed)
#		print('Seed: ' + str(seed))
		
		print()
		prog_bar = tqdm(total=self.max_nr_legs, leave=True, position=0)
		
		self.aco.initialize()
		stats = self.aco.path.stats
		
		while (self.max_nr_legs is None) or (stats.nr_legs < self.max_nr_legs):
			self.aco.build_generation()
#			self.print_best()
			self.aco.nr_gen += 1
			prog_bar.desc = self.stats_best() + ' '
			prog_bar.update(stats.nr_legs - prog_bar.n)
			if self.max_nr_gens == self.aco.nr_gen:
				break
		
		prog_bar.desc = ''; prog_bar.refresh();
		prog_bar.close()
		
	
	def start(self):
		"Conduct an experiment, by performing multiple independent runs."
		self.show_setup()
		
		stats, trajs = [], []
		fname = self.path + self.filename
		
#		for r in range(self.nr_runs):
		for r in trange(self.nr_runs, leave=True, desc='RUNS'):
			
			self.run(seed=r)
			self.print_best()
			
			# save experimental data
			stats.append(self.aco.path.stats.export())
			trajs.append(self.aco.best)
			if (r + 1) % self.log_data_every == 0:
				safe_dump(stats, fname, append=True)
				safe_dump(trajs, fname[:-3] + 'TRAJ.pkl', append=True)
				stats, trajs = [], []
		
		if stats != []:
			safe_dump(stats, fname, append=True)
			safe_dump(trajs, fname[:-3] + 'TRAJ.pkl', append=True)
		
	

# ==================================== ## ==================================== #
# ------------------------------------ # Launching experiments

def bs_cost(bw, bf, nodes=1, depth=1, max_depth=16, total=1):
	"""
	Estimate the number of leg evaluations performed in a Beam Search,
	with a given beam width (`bw`) and branch factor (`bf`), to
	reach a score of `max_depth`.
	"""
	if depth == max_depth:
		return total
	n = nodes * bf
	total += n
	n = min(n, bw)
	return bs_cost(bw, bf, n, depth + 1, max_depth, total)
	


def go(variant, multiobj=True, path=None, **kwargs):
	"Launch an experiment"
	
	# define path where experimental results will be saved
	if path is None:
		path = 'results/traj_search/'
	if variant == 'Beam Search':
		path += 'Beam Search '
		path += '(multi obj)/' if multiobj else '(single obj)/'
	
	# configure the number of runs, and stopping criterion
	if variant == 'Beam Search':
		exp_args = dict(
			nr_runs=1, log_data_every=1, max_nr_legs=None, max_nr_gens=1)
	else:
		exp_args = dict(nr_runs=100, log_data_every=2, max_nr_legs=100000)
	
	# instantiate path handler
	# (Pareto Elitism used in the search method if path handler sorts solutions
	# based on Pareto dominance)
	ph = init__path_handler(multiobj_evals=multiobj)
	exp_args.update(dict(path_handler=ph, pareto_elite=multiobj))
	
	exp_args.update(kwargs)
	
	# RUN experiment
	e = experiment(variant=variant, path=path, **exp_args)
	e.start()
	return e
	


# ------------------------------------ # Experimental plan

if __name__ == "__main__":
	
	
#	go('P-ACO', alpha=1, beta=1, multiobj=True)
	
	
	args = dict(multiobj=True, beam_width=5, branch_factor=75)
	go('Stochastic Beam', **args)
	go('Beam P-ACO', alpha=1, beta=1, **args)
	
	args = dict(multiobj=True, beam_width=5, branch_factor=250)
	go('Stochastic Beam', **args)
	go('Beam P-ACO', alpha=1, beta=1, **args)
	
	args = dict(multiobj=True, beam_width=10, branch_factor=250)
	go('Stochastic Beam', **args)
	go('Beam P-ACO', alpha=1, beta=1, **args)

#	args = dict(multiobj=True, beam_width=15, branch_factor=125)
#	go('Stochastic Beam', **args)
#	go('Beam P-ACO', alpha=1, beta=1, **args)
	
#	args = dict(multiobj=True, beam_width=15, branch_factor=175)
#	go('Stochastic Beam', **args)
#	go('Beam P-ACO', alpha=1, beta=1, **args)
	
	args = dict(multiobj=True, beam_width=20, branch_factor=125)
	go('Stochastic Beam', **args)
	go('Beam P-ACO', alpha=1, beta=1, **args)
	
#	args = dict(multiobj=True, beam_width=20, branch_factor=250)
#	go('Stochastic Beam', **args)
#	go('Beam P-ACO', alpha=1, beta=1, **args)
	
	args = dict(multiobj=True, beam_width=25, branch_factor=50)
	go('Stochastic Beam', **args)
	go('Beam P-ACO', alpha=1, beta=1, **args)
	
	
#	""" (deterministic) Beam Search runs
#	for bw in [5, 10, 15, 20]:
#	for bw in [25, 30, 35, 40, 45, 50]:
	for bw in range(5, 50+1, 5):
		for bf in range(25, 500+1, 25):
			# only run setups where 100k legs are sufficient to reach score 16
			if bs_cost(bw, bf) > 100000:
				continue
			go('Beam Search', multiobj=True, beam_width=bw, branch_factor=bf)
#	"""	
	
