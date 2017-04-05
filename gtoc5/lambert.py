# 
# Copyright (c) 2017 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the MIT License. See the LICENSE file for details.


from math import sqrt, exp
from functools import wraps, partial #, total_ordering

import numpy as np
import PyKEP as pk
from scipy.optimize import minimize_scalar

from .constants import MU_SUN, G0, SEC2DAY, DAY2SEC, I_sp, T_max, thrust_tol
from .multiobjective import pareto_front



# ==================================== ## ==================================== #

class obj_value(float):
	"""
	Subclass of `float` that takes its value from a	newly instantiated object's
	`.get_value()` method. The instantiated object is made available in `.obj`.
	
	Allows for arithmetic and comparison operations over the object to be
	redirected to the methods of `float`, acting on the object's reference
	value, without the need to define such methods in the object's class.
	"""
	def __new__(cls, obj_class, *args, **kwargs):
		obj = obj_class(*args, **kwargs)
		self = float.__new__(cls, obj.get_value())
		self.obj = obj
		return self


def expand_kwargs(func):
	"""
	Allows for kwargs that are to be sent to a function to be provided within a
	dictionary sent as the last *arg. Useful when a function's caller can be set
	to redirect *args, but not *kwargs (e.g.: np.apply_along_axis, until 1.9.0).
	"""
	@wraps(func)
	def wrapper(*args, **kwargs):
		if isinstance(args[-1], dict):
			return func(*args[:-1], **args[-1])
		else:
			return func(*args, **kwargs)
	return wrapper



#@total_ordering
class lambert_eval(object):
	
	@expand_kwargs
	def __init__(self, leg_dT, dep_ast, arr_ast, dep_t, dep_m, *args, **kwargs):
		self.dT = leg_dT
		self.dep_ast = dep_ast
		self.arr_ast = arr_ast
		self.dep_t = dep_t
		self.arr_t = self.dep_t + self.dT
		self.dep_m = dep_m
		
		s = self.solve(*args, **kwargs)
		if s is not None:
			self.select(*(s + args), **kwargs)
			self.inspect(*args, **kwargs)
		
	
	def get_value(self):
		"Key value determining the instance's solution quality."
		return self.dV
		
	
	def solve(self, *args, validate_barker=True, verbose=False, **kwargs):
		"Solves Lambert's problem for the requested transfer."
		
		# departure and arrival epochs
		dep_t = pk.epoch(self.dep_t, 'mjd')
		arr_t = pk.epoch(self.arr_t, 'mjd')
		
		# departure and arrival positions & velocities
#		/-> (could obtain `dep_ast_eph` in `lambert_optimize_dt` and pass it as
#		|   argument to avoid having it being repeatedly calculated here)
#		r1, v1 = self.dep_ast.eph(dep_t) if dep_ast_eph is None else dep_ast_eph
		r1, v1 = self.dep_ast.eph(dep_t)
		r2, v2 = self.arr_ast.eph(arr_t)
		
		# Barker equation used to skip useless Lambert computations
		# https://en.wikipedia.org/wiki/Parabolic_trajectory#Barker.27s_equation
		if validate_barker and self.dT < pk.barker(r1, r2, MU_SUN) * SEC2DAY:
			if verbose:
				print(self.dT, 'Fails Barker:',
				      self.dT, pk.barker(r1, r2, MU_SUN) * SEC2DAY)
			self.fail()
			return None
		
		l = pk.lambert_problem(r1, r2, self.dT * DAY2SEC, MU_SUN)
		# don't compute any multi-rev solutions:
		#l = pk.lambert_problem(r1, r2, self.dT * DAY2SEC, MU_SUN, False, 0)
		
		return l, v1, v2
		
	
	def select(self, lamb_sol, v_body1, v_body2, *args, **kwargs):
		"""
		Selects one of the Lambert's problem solutions
		(in case multiple revolution solutions were found).
		Selection criterion: solution with the smallest dV.
		"""
		# get, per solution, the spacecraft's velocity at each body
		v1sc = lamb_sol.get_v1()
		v2sc = lamb_sol.get_v2()
		
		# determine each solution's dV
		solutions = []
		for v1, v2 in zip(v1sc, v2sc):
			dV1 = sqrt(sum((a - b) * (a - b) for (a, b) in zip(v_body1, v1)))
			dV2 = sqrt(sum((a - b) * (a - b) for (a, b) in zip(v_body2, v2)))
			solutions.append((dV1 + dV2, v1, v2))
		
		# pick the solution with smallest dV, and log the spacecraft's
		# velocities at each body
		self.dV, *self.v_sc = min(solutions)
		
	
	def inspect(self, validate_acc=True, verbose=False, *args, **kwargs):
		"Validation and post-processing of the selected solution."
		
		if validate_acc:
			# feasibility check on the acceleration
			leg_accel = self.dV / (self.dT * DAY2SEC)
			max_accel = thrust_tol * T_max / self.dep_m
			
			if leg_accel >= max_accel:
				if verbose:
					print(self.dT, 'Fails Accel.:', leg_accel, max_accel)
				self.fail()
				return
		
		self.feasible = True
		
		# get the arrival mass (kg), given mass at departure (kg) and dV (m/s)
		# https://en.wikipedia.org/wiki/Tsiolkovsky_rocket_equation
		self.arr_m = self.dep_m * exp(self.dV / (-I_sp * G0))
		
	
	def fail(self):
		"Signals the failure to identify a feasible solution."
#		self.dV = float('inf')
		self.dV = 100000.
		self.v_sc = None
		self.feasible = False
		self.arr_m = None
		
	
#	def __lt__(self, other):
#		return self.get_value() < other.get_value()
#	
#	def __eq__(self, other):
#		return self.get_value() == other.get_value()
		
	


# ==================================== ## ==================================== #

def lambert_optimize_dt(dep_ast, arr_ast, dep_t, dep_m,
	                    leg_dT=None, leg_dT_bounds=None, nr_evals=50,
	                    obj_fun=lambert_eval, grid=True,
	                    random_pareto=False, random=None,
	                    **kwargs):
	"""
	Find the transfer time between two given asteroids that optimizes the
	given objective function.
	
	Parameters:
	-----------
	dep_ast
		departure asteroid
	arr_ast
		arrival asteroid
	dep_t
		time of departure (mjd) from asteroid `dep_ast`
	dep_m
		spacecraft's mass (kg) at departure
	leg_dT
		an exact leg_dT to be used in the Lambert arc
		(if specified, no optimization is then performed over leg_dT)
	leg_dT_bounds
		bounds on the time of flight
		(used if `leg_dT=None`)
	nr_evals
		number of solutions to evaluate
		(an exact amount if doing grid search; an upper bound if optimizing)
	obj_fun
		objective function that creates & evaluates Lambert arcs (goal: min)
	grid
		if True: finds best leg_dT over an evenly spaced grid of options
		if False: optimizes leg_dT with minimize_scalar()
	random_pareto
		indication of whether to choose the transfer time by picking a random
		solution from the Pareto front of solutions in the grid
		(requires `grid=True`)
	random
		random number generator to use, if `random_pareto=True`
		(defaults to np.random)
	**kwargs
		extra arguments to be sent to `obj_fun`
	"""
	assert (leg_dT_bounds is None) ^ (leg_dT is None), 'One (and only one) of' \
		' leg_dT_bounds or leg_dT must be specified.'
	assert not random_pareto or (random_pareto and grid), 'random_pareto=True' \
		' requires grid=True.'
	
	if leg_dT is not None:
		grid = True
		random_pareto = False
#	if grid and random_pareto and random is None:
#		random = np.random
#	\-> (for call to be reproducible, caller must specify a seeded `random`)
	
	# wrap object created by `obj_fun` in a `obj_value` instance, making the
	# objective value under optimization more accessible and simpler to handle
	obj_fun = partial(obj_value, obj_fun)
	
	# extra arguments to be sent to `obj_fun`
	eval_args = (dep_ast, arr_ast, dep_t, dep_m, kwargs)
	
	if grid:
		# Generate `nr_evals` evenly spaced values in between the leg dT bounds.
		# Unless: if a specific leg_dT is provided, only that one is attempted.
		if leg_dT is None:
			leg_dTs = np.linspace(*leg_dT_bounds, num=nr_evals)
		else:
			leg_dTs = [leg_dT]
		
		grid = (obj_fun(leg_dT, *eval_args) for leg_dT in leg_dTs)
		
		if not random_pareto:
			# obtain the point in the grid with minimal cost value
			best = min(grid).obj
		else:
			# pick a random point from the the Pareto front of trade-offs
			# between transfer time and leg cost
			grid_l = list(grid)
			grid = [t.obj for t in grid_l if t.obj.feasible]
			if grid == []:
				return grid_l[0].obj
			trade_offs = [(t.dT, t.get_value()) for t in grid]
			pf = pareto_front(trade_offs)
			best = grid[random.choice(pf)]
	else:
		# minimize scalar function of one variable
		# https://docs.scipy.org/doc/scipy/reference/optimize.html
		best = minimize_scalar(obj_fun, args=eval_args,
		                       method='bounded', bounds=leg_dT_bounds,
		                       options=dict(maxiter=nr_evals)
		                       ).fun.obj
	
	return best
	
