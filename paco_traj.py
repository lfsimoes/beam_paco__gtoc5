"""
Single and multi-objective path handlers for interfacing the P-ACO/Beam Search
algorithms with the GTOC5 problem.

Also provides functionality for tracking statistics about the search for
trajectories.
"""
# 
# Copyright (c) 2017 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the MIT License. See the LICENSE file for details.


from functools import lru_cache
from itertools import combinations
from collections import namedtuple, Counter

from paco import *

from gtoc5 import *
from gtoc5.phasing import rate__euclidean, rate__orbital, rate__orbital_2
from gtoc5.multiobjective import *



# ==================================== ## ==================================== #
# ------------------------------------ # Run statistics

quality_nt = namedtuple('quality', ['score', 'agg', 'mass', 'tof'])
when_nt = namedtuple('when', [
	'nr_gen',
	'nr_traj',
	'nr_legs',
	'nr_legs_distinct',
	'nr_legs_feasible',
	'nr_lambert'])


class traj_stats(object):
	
	def __init__(self, aco=None, path_handler=None):
		# links to the search algorithm & path handler instances
		self.aco = aco
		self.path = path_handler
		
		self.nr_traj = 0
		self.nr_legs = 0
		self.nr_legs_distinct = 0
		self.nr_legs_feasible = 0
		self.nr_lambert = 0
		
		self.when = []
		self.seq = []
		self.quality = []
		
	
	def log_mission(self, m):
		"log statistics from a completed trajectory"
		self.nr_traj += 1
		
		_score = score(m)
		_agg = resource_rating(m)
		_mass = final_mass(m)
		_tof = tof(m) * DAY2YEAR
		
		t = (self.aco.nr_gen, self.nr_traj, self.nr_legs,
		     self.nr_legs_distinct, self.nr_legs_feasible, self.nr_lambert)
		
		self.when.append(t)
		self.seq.append(seq(m, incl_flyby=False))
		self.quality.append((_score, _agg, _mass, _tof))
		
	
	def export(self):
		return (self.when, self.seq, self.quality)
		
	
	def load(self, *args):
		self.nr_traj, self.nr_lambert = None, None
		self.nr_legs, self.nr_legs_distinct, self.nr_legs_feasible = [None]*3
		self.when, self.seq, self.quality = args
		return self
		
	
	def when_threshold_reached(self, max_nr_legs=100000):
		"""
		Determine the index into when/seq/quality corresponding to the last
		time instant in which the algorithm was still within a certain budget of
		number of legs produced
		"""
		max_i = None
		for (i, w) in enumerate(self.when):
			if w[2] >= max_nr_legs:
				break
			max_i = i
		return max_i
		
	
	def when_best_found(self, max_i=None):
		"""
		Index into the instant at which the best solution was found.
		(solution with greatest score *and* resource rating)
		"""
		max_q = None
		max_qi = None
		for (i, q) in enumerate(self.quality[:max_i]):
			if max_q is None or q[:2] > max_q[:2]:
				max_q = q
				max_qi = i
		return max_qi
		
	
	def best(self, max_i=None, max_qi=None):
		"Quality of the run's best solution."
		if max_qi is None:
			max_qi = self.when_best_found(max_i)
		return quality_nt(*self.quality[max_qi])
		
	
	def best_seq(self, max_i=None, max_qi=None):
		"Sequence of asteroids in the run's best found solution."
		if max_qi is None:
			max_qi = self.when_best_found(max_i)
		return self.seq[max_qi]
		
	
	def evals_to_best(self, max_i=None, max_qi=None):
		"""
		Tuple describing the instant in which the run's best solution
		was found.
		"""
		if max_qi is None:
			max_qi = self.when_best_found(max_i)
		return when_nt(*self.when[max_qi])
		
	
	def evals_to_score(self, score, last=False, max_i=None):
		"""
		Tuple describing the instant in which the first (or last) solution
		with a given `score` (or better) was found.
		"""
		i = None
		for (w, q) in zip(self.when[:max_i], self.quality[:max_i]):
			if q[0] >= score:
				i = when_nt(*w)
				if not last:
					break
		return i
		
	
	def seq_similarity(self, discard_first=2, sample_size=10000, max_i=None):
		"""
		Measure of similarity among all asteroid sequences produced in a run.
		Ignores sequences' first two bodies (Earth + 1st asteroid), by default,
		as these necessarily repeat across generated trajectories.
		"""
		return seq_similarity(self.seq[:max_i], discard_first, sample_size)
		
	
	def cumul_nr_seqs(self, score, count_distinct=True, discard_first=1):
		"""
		Determine the cumulative number of sequences found over time, distinct
		or not, with at least a given `score`.
		"Time" is given by the total number of legs obtained to date.
		"""
		tmax = self.when[-1][2]
		c = np.zeros(tmax + 1)
		seen_seqs = set()
		
		for w, s, q in zip(self.when, self.seq, self.quality):
			if q[0] < score:
				continue
			t = w[2]
			if not count_distinct:
				c[t:] += 1
			else:
				# discard non-scoring ids at the start of the sequence,
				# and keep only the `score` first elements
				seq = tuple(s[discard_first : score + discard_first])
				if seq not in seen_seqs:
					c[t:] += 1
					seen_seqs.add(seq)
		
		return c
		
	
	def quality__multiobj_fmt(self):
		"""
		Returns a conversion of the .quality list, where evaluations are
		formatted as a tuple containing:
		1. score (number of asteroids fully explored in the mission)
		2. fuel mass consumed [kg]
		3. total flight time [years]
		The ideal mission will have MAX score, with MIN mass and time costs.    
		(this is the same format used in `class gtoc5_ant_pareto / evaluate()`)
		"""
		return [
			(_score, MASS_MAX - _mass, _tof)
			for (_score, _agg, _mass, _tof) in self.quality
			]
		
	
	def pareto_front_evolution(self, every_nr_legs=1000, up_to_nr_legs=100000,
	                           if_score=16):
		"""
		Determine the Pareto fronts of solutions generated up to different
		points in time.
		"""
		def log_pareto_front():
			ev = list(evals_bin)
			pf = [ev[i] for i in pareto_front(ev)]
			bins.append((t_thresh, pf))
			return pf
		
		bins = []
		evals_bin = set()
		t_thresh = every_nr_legs
		
		for t, q in zip(self.when, self.quality__multiobj_fmt()):
			
			if t[2] > t_thresh: # current nr_legs exceeds time threshold? 
				pf = log_pareto_front()
				evals_bin = set(pf)
				t_thresh += every_nr_legs
			if t_thresh > up_to_nr_legs:
				break
			
			if q[0] == if_score:
				evals_bin.add(q[1:])
		
		if t_thresh <= up_to_nr_legs:
			# loop was interrupted by exhaustion of element to iterate over.
			# force one last addition to `bins()`, to consider elements that may
			# have been added to `evals_bin` since the last update.
			log_pareto_front()
		
		if bins == []:
			return [], []
		time_threshold, _pareto_front = zip(*bins)
		return time_threshold, _pareto_front
		
	
	def hypervolume_evolution(self, pf_evol=None, **kwargs):
		"""
		Determine hypervolumes of the Pareto fronts of solutions generated up to
		different points in time.
		"""
		refpt = (MASS_MAX - MASS_MIN, TIME_MAX * DAY2YEAR) # (3500 kg, 15 years)
		if pf_evol is None:
			pf_evol = self.pareto_front_evolution(**kwargs)
		t, pf = pf_evol
		return t, [hypervolume(f, refpt) for f in pf]
		
	

# ==================================== ## ==================================== #
# ------------------------------------ # Sequence similarity measures

def seq_similarity(seqs, discard_first=0, sample_size=None):
	"""
	Measure of the similarity among asteroid sequences (regardless of order).
	Optionally discards an initial sub-sequence of asteroids that may
	necessarily repeat across sequences (`discard_first`).
	Considers all possible pairs of sequences in `seq`, unless a `sample_size`
	is provided, in which case `sample_size` random pairings of sequences will
	be used instead to estimate the similarity.
	
	Returns mean & st. dev. of a value in [0, 1].
	* 0: no asteroid ever occurs in more than one sequence;
	* 1: all given sequences contain exactly the same asteroids.
	
	Note: among two given sequences, the maximum number of asteroids that may
	repeat is bounded by the size of the smallest sequence.
	"""
	common_ratio = []
	
	if sample_size is None:
		pairs = combinations(seqs, 2)
	else:
		pairs = (
			(seqs[a], seqs[b if b < a else b + 1])
			for a in np.random.randint(0, len(seqs), size=sample_size)
			for b in np.random.randint(0, len(seqs) - 1, size=1)
			)
	
	for a, b in pairs:
		a = set(a[discard_first:])
		b = set(b[discard_first:])
		
		# calculate the fraction of asteroids common to both sequences
		# (the maximum is bounded by size of the smallest sequence)
		min_len = min(len(a), len(b))
		if min_len == 0:
			c = 0.
		else:
			c = len(a & b) / min_len
		
		common_ratio.append(c)
	
	return np.mean(common_ratio), np.std(common_ratio)
	


def ast_frequency_dist(seqs, discard_first=0):
	"""
	Distribution of asteroid frequencies.
	List of (x, y), where:
		x: number of sequences (among `seq`) that contain a given asteroid;
		y: number of distinct asteroids with that level of occurrence.
	Optionally discards an initial sub-sequence of asteroids that may
	necessarily repeat across sequences (`discard_first`).

	Examples
	--------
	>>> ast_frequency_dist([(1,2,3), (3,4,5,6), (1,3,4)])
	[(1, 3), (2, 2), (3, 1)]
	# 3 asteroids appear in only one sequence, 2 in 2, and 1 in all 3 (ast. 3)
	"""
	ast_freq = Counter([a for s in seqs for a in s[discard_first:]])
	return sorted(Counter(ast_freq.values()).items())
	


# ==================================== ## ==================================== #
# ------------------------------------ # Heuristic functions

def heuristic(rating, gamma=50, tabu=()):
	"""
	Converts the cost ratings for a group of trajectory legs (min is better)
	into a selection probability per leg (max is better).
	
	The greater the provided `gamma` exponent, the greedier the probability
	distribution will be, in favoring the best rated alternatives.
	
	Alternatives at the `tabu` indices will get a selection probabity of 0.0.
	"""
	# Rank the cost ratings. Best solutions (low cost) get low rank values
	rank = np.argsort(np.argsort(rating))
	# scale ranks into (0, 1], with best alternatives going from
	# low ratings/costs to high values (near 1).
	heur = 1. - rank / len(rank)
	# scale probabilities, to emphasize best alternatives
	heur = heur**float(gamma)
	# assign 0 selection probability to tabu alternatives
	heur[tuple(tabu),] = 0.0
	return heur
	


def heuristic__norml(rating, gamma=50, tabu=()):
	"""
	Converts the cost ratings for a group of trajectory legs (min is better)
	into a selection probability per leg (max is better).
	
	The greater the provided `gamma` exponent, the greedier the probability
	distribution will be, in favoring the best rated alternatives.
	
	Alternatives at the `tabu` indices will get a selection probabity of 0.0.

	||| Alternative to the `heuristic()` function, that normalizes ratings based
	||| on their extreme values. The resulting heuristic values are therefore 
	||| sensitive to rating outliers.
	"""
	tabu = tuple(tabu)
	
	# build a mask over ratings to considered (the non-tabu ones)
	incl_mask = np.ones(len(rating), dtype=np.bool)
	incl_mask[tabu,] = False
	
	# scale ratings into [0, 1], with best alternatives going from
	# low ratings/costs to high values (near 1).
	m = np.min(rating[incl_mask])
	M = np.max(rating[incl_mask])
	heur = 1.0 - (rating - m) / (M - m)
	
	# assign 0 selection probability to tabu alternatives
	heur[tabu,] = 0.0
	
	# scale probabilities, to emphasize best alternatives
	heur = heur**float(gamma)
	
	return heur
	


# ==================================== ## ==================================== #
# ------------------------------------ # Path handler for GTOC5 missions

# The classes below implement the same interface as paco.py's `tsp_path`
# class, which allows instances of `paco` or its sub-classes to solve
# Travelling Salesman Problem instances.

class gtoc5_ant(object):
	"""
	Handler for GTOC5 trajectories subjected to single-objective evaluations.
	"""
	
	# indication that the costs of asteroid transfers are *not* symmetric
	symmetric = False
	
	# indication that the path handler, via the `.tabu()` method, *disallows*
	# previously visited nodes/asteroids from being revisited.
	allows_revisits = False
	
	
	def __init__(self, starting_ast=1712, ratingf=None, ref_dT=125, gamma=50,
	             add_ast_args=None):
		# initial mission specification from which all ant walks will start
		self.start_conditions = mission_to_1st_asteroid(starting_ast)
		# phasing indicator used to rate destination asteroids
		self.rate_destinations = rate__orbital_2 if ratingf is None else ratingf
		# reference transfer time used in the Orbital phasing indicator
		self.ref_dT = ref_dT
		# "greediness exponent" used in the `heuristic` function
		self.gamma = gamma
		# parameters configuring the addition of new legs to missions.
		# if unspecified, defaults throughout the called functions to:
		# dict(leg_dT_bounds=rvleg_dT_bounds, nr_evals=50,
		#      obj_fun=gtoc5_rendezvous, grid=True, random_pareto=False)
		self.add_ast_args = {} if add_ast_args is None else add_ast_args
		
	
	def initialize(self, aco):
		"ACO is starting a new run. Reset all run state variables."
		self.add_ast_args['stats'] = self.stats = traj_stats(aco, self)
		self.add_ast_args['random'] = aco.random # needed if random_pareto=True
		# reset the caches
		self._cached_heuristic.cache_clear()
		reset_leg_cache()
		
	
	def heuristic(self, ant_path):
		"Heuristic used to estimate the quality of node transitions."
		dep_ast = ant_path[-1][0]
		dep_t = ant_path[-1][2]
		tabu = tuple(self.tabu(ant_path))
		return self._cached_heuristic(dep_ast, dep_t, tabu)
	
	@lru_cache(maxsize=2**14)
	def _cached_heuristic(self, dep_ast, dep_t, tabu):
		rating = self.rate_destinations(dep_ast, dep_t, leg_dT=self.ref_dT)
		# tabu: ids of bodies already visited get a 0 selection probability
		return heuristic(rating, self.gamma, tabu)
		
	
	def start(self):
		"Start a new path through the graph."
		self.stop_walk = False
		return self.start_conditions.copy()
		
	
	def tabu(self, ant_path):
		"List of nodes to exclude from consideration as future nodes to visit."
		# revisits are disallowed, as a visit already fully scores the asteroid.
		# all previously visited asteroids/nodes are then tabu.
		return self.get_nodes(ant_path)
		
	
	def add_node(self, ant_path, node):
		"Extend an ant's path with a new visited node."
		success = add_asteroid(ant_path, int(node), **self.add_ast_args)
		self.stop_walk = not success
		
	
	def get_nodes(self, ant_path):
		"Get the list of nodes visited so far along the ant's path."
		return seq(ant_path, incl_flyby=False)
		
	
	def get_links(self, ant_path):
		"Get an iterator over node transitions performed along an ant's path."
		path_nodes = self.get_nodes(ant_path)
		return zip(path_nodes[:-1], path_nodes[1:])
		
	
	def stop(self, ant_path, force_stop=False):
		"Indicate whether an ant's path should be terminated."
		# stops walk/trajectory if the lastest attempt to visit a new asteroid
		# failed (see `.add_node()`)
		if self.stop_walk or force_stop:
			self.stats.log_mission(ant_path)
			return True
		return False
		
	
	def evaluate(self, ant_path):
		"Quality function used to evaluate an ant's path through the graph."
		# score + "resource savings" rating
#		return score(ant_path) + resource_rating(ant_path)
		
		# score + fraction of usable mass left
#		return score(ant_path) + resource_rating(ant_path, ret_criteria=True)[1]
		m = final_mass(ant_path)
		frac_mass_available = (m - MASS_MIN) / (MASS_MAX - MASS_MIN)
		return score(ant_path) + frac_mass_available
		
	
	def sort(self, evaluated_paths, r=None):
		"""
		Given a list of `evaluated_paths` (a list of (quality, ant_path)
		tuples), return a ranked list, with the top `r` paths (or all, if
		unspecified), sorted by decreasing order of quality
		(decreasing order of score + resource availability).
		"""
		if r == 1:
			return [max(evaluated_paths, key=lambda i: i[0])]
		return sorted(evaluated_paths, key=lambda i: i[0], reverse=True)[:r]
		
	
	def copy(self, ant_path):
		"Create a copy of a given ant path."
		return ant_path.copy()
		
	

class gtoc5_ant_pareto(gtoc5_ant):
	"""
	Handler for GTOC5 trajectories subjected to multi-objective evaluations.
	
	Specialization of the `gtoc5_ant` class that uses Pareto dominance
	to evaluate and sort ant paths (missions).
	"""
	
	def initialize(self, aco):
		super(gtoc5_ant_pareto, self).initialize(aco)
#		self.random = aco.random
		
	
	def evaluate(self, ant_path):
		"""
		Quality function used to evaluate an ant's path through the graph.
		
		Produces an evaluation according to multiple criteria:
		1. score (number of asteroids fully explored in the mission)
		2. fuel mass consumed [kg]
		3. total flight time [years]
		
		The ideal mission will have MAX score, with MIN mass and time costs.
		"""
		mass_used = MASS_MAX - final_mass(ant_path)
		time_used = tof(ant_path) * DAY2YEAR
		return (score(ant_path), mass_used, time_used)
		
	
	def sort(self, evaluated_paths, r=None, f=None):
		"""
		Given a list of `evaluated_paths` (a list of (quality, ant_path)
		tuples), return a list with paths sorted by decreasing order of quality.
		
		Uses Pareto dominance to sort paths:
		* higher scoring missions dominate lower scoring ones, and so will
		  precede those in the sorted list;
		* among equally scored missions, non-dominated sorting will be used
		  to group missions with mutually non-dominating mass and time costs;
		* missions in the same front are sorted by ascending order of the first
		  (minimization) criterion: mass used.
		
		A subset containing a number of top solutions will be provided instead
		if requested via the arguments:
		* `r` (rank threshold): returns only the top `r` solutions;
		* `f` (front threshold): returns only the top `f` Pareto fronts.
		"""
		# separate paths in bins, gathering all those that share the same score
		score_bins = {}
		for qp in evaluated_paths:
			score = qp[0][0]
			score_bins.setdefault(score, []).append(qp)
		
		# If only one solution (or Pareto front) is requested, keep only
		# the highest scored missions, and discard the remaining ones.
		# Otherwise, sort bins by decreasing order of score.
		if r == 1 or f == 1:
			evaluated_paths = [max(score_bins.items())[1]]
		else:
			score_bins = sorted(score_bins.items(), reverse=True)
			evaluated_paths = [ev_paths for (score, ev_paths) in score_bins]
		
		# get, per score bin in `evaluated_paths`, the list of objective pairs
		# evaluating the mass and time of each mission in that bin
		objs = [
			[q[1:] for (q, p) in score_bin]
			for score_bin in evaluated_paths]
		
		# If only one solution (or Pareto front) is requested, obtain the
		# Pareto front of the highest scored paths. Then, as requested, either
		# return that whole front, or a single path chosen from it.
		if r == 1 or f == 1:
			o, p = objs[0], evaluated_paths[0]
			pf = pareto_front(o)
			if f == 1:
				return [p[idx] for idx in pf]
			else:
#				# pick a random path from the Pareto front
#				return [p[self.random.choice(pf)]]
#				# pick the path with highest resource rating
#				return [max(p, key=lambda i: resource_rating(i[1]))]
				# pick the first point in the Pareto front (min mass cost)
				return [p[pf[0]]]
		
		ranked_paths = []
		# number of paths, and fronts, fed into `ranked_paths`
		r_len = 0
		f_len = 0
		
		# iterate over groups/bins of paths having the same score
		for obj, ev_paths in zip(objs, evaluated_paths):
			
			# peform non-dominated sorting of the considered paths
			nds = non_dominated_sorting(obj)
			
			for pf in nds:
				# if adding paths from the current front would bring us over the
				# requested number of paths, fill up the ranked list with a
				# subset of paths from the front
				if r is not None and r_len + len(pf) > r:
#					# pick a random subset of paths from the Pareto front
#					pf = self.random.choice(pf, size=r - r_len, replace=False)
#					# pick the paths with highest resource rating
#					pf.sort(
#						key=lambda i: resource_rating(ev_paths[i][1]),
#						reverse=True)
#					pf = pf[:r - r_len]
					# pick the paths with smallest mass cost (slicing is
					# sufficient, as `non_dominated_sorting` returns indices
					# lexicographically sorted by ascending order of cost)
					pf = pf[:r - r_len]
				
				ep = [ev_paths[idx] for idx in pf]
				
				ranked_paths.extend(ep)
				r_len += len(ep)
				f_len += 1
				
				# if we've reached the requested number of paths (or fronts),
				# return the paths we've ranked and accumulated so far
				if r is not None and r_len == r:
					return ranked_paths
				if f is not None and f_len == f:
					return ranked_paths
		
		# If we've reached this point, all the provided `evaluated_paths` are
		# now ranked in `ranked_paths`. Either `r` and `f` were not specified,
		# or the provided paths didn't allow for those thresholds to be reached
		return ranked_paths
		
	
