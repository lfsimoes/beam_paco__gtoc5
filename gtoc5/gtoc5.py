# 
# Copyright (c) 2017 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the MIT License. See the LICENSE file for details.


from math import sqrt, log
import os, inspect, pickle

import numpy as np
import PyKEP as pk

from .constants import *
from .lambert import lambert_eval, lambert_optimize_dt
from .multiobjective import rate_traj



# ==================================== ## ==================================== #

def seq(mission, incl_flyby=True):
	"Sequence of asteroids visited in the mission."
	return [mission[0][0]] + \
	       [l[0] for l in mission[1:]][::1 if incl_flyby else 2]


def final_mass(mission):
	"Final mass, after completing the last leg."
	return mission[-1][1]


def tof(mission):
	"Time of flight. Duration, in days, of the complete mission."
	return mission[-1][2] - mission[0][2]


def resource_rating(mission, **kwargs):
	"Resource savings rating (softmin aggregation)."
	return rate_traj(final_mass(mission), tof(mission), **kwargs)


def score(mission):
	"Calculate the mission's score."
	asts_rv = set()
	asts_rvfb = set()
	score = 0.0
	for ast in seq(mission, incl_flyby=True)[1:]: # [1:] skips the Earth
		sc = 0.0
		if ast in asts_rv and ast not in asts_rvfb:
			sc = 0.8			# flyby score
			asts_rvfb.add(ast)
		elif ast not in asts_rv:
			sc = 0.2			# rendezvous score
			asts_rv.add(ast)
		if ast == 1:
			sc *= 1.5			# bonus for the Beletskij asteroid
		score += sc
	return score



# ==================================== ## ==================================== #

def mission_to_1st_asteroid(ast1, legs1=None):
	"""
	Initialize a `mission` data structure containing a leg from Earth to the
	first asteroid, extended with a self-flyby of the asteroid.
	Rendezvous leg uses data obtained from a low-thrust global optimization,
	and self-flyby leg determined via the linear acceleration model.
	"""
	# Summary data for the launch leg loaded from the results of
	# mass/time-optimal low-thrust global optimizations described in:
	# - http://dx.doi.org/10.2420/AF08.2014.45 (Sec. 2)
	# - https://github.com/esa/pagmo/blob/master/src/problem/gtoc5_launch.cpp
	if legs1 is None:
		# get path to where the current module is located
		path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda:0)))
#		legs1 = pickle.load(open(path + '/mass_optimal_1st_leg.pkl', 'rb'))
		legs1 = pickle.load(open(path + '/time_optimal_1st_leg.pkl', 'rb'))
	
	try:
		# locate in `legs1` the tuple corresponding to the leg towards `ast1`
		leg1 = next(ast_leg for ast_leg in legs1 if ast_leg[0] == ast1)
	except StopIteration:
		raise Exception('No known launch leg towards asteroid %s (id: %d)' % (
			asteroids[ast1].name, ast1)) from None  # (PEP 409)
	
	# get the first leg's parameters
	dep_m = MASS_MAX
	(ast1, arr_m, dep_t, arr_t) = leg1
	
	# add launch
	earth_id = 0
	mission = [(earth_id, dep_m, dep_t, 0.0, 0.0)]
	
	# add rendezvous leg
	mass_rv = arr_m - MASS_EQUIPMENT
	dT_rv = arr_t - dep_t
	dV_rv = I_sp * G0 * log(dep_m / arr_m)
	
	mission.append(
		(	ast1,			# asteroid UID
			mass_rv,		# mass at asteroid, after the payload delivery
			arr_t,			# Epoch at time of departure to self-flyby
			dT_rv,			# leg dT
			dV_rv			# leg dV
			) )
	
	# add self-flyby leg
	mission.append(self_flyby_leg(mission))
	
	return mission
	


# ==================================== ## ==================================== #

LEG_CACHE = {}

def reset_leg_cache():
	global LEG_CACHE
	LEG_CACHE = {}
	

def add_asteroid(mission, next_ast, use_cache=True, stats=None, **kwargs):
	"""
	Extend `mission` by visiting a new asteroid.
	Adds rendezvous and self-flyby legs, thus fully scoring the asteroid.
	"""
	global LEG_CACHE
	
	assert isinstance(next_ast, (int, np.integer)) and 0 < next_ast <= 7075, \
		"Next asteroid should be given as an integer in {1, ..., 7075}."
	next_ast = int(next_ast)
	
	if stats is not None:
		# increment total number of [rendezvous] legs defined, either from
		# a new optimization, or from a cache hit. Ignores feasibility.
		# (will be equal to stats.nr_legs_distinct if use_cache==False)
		stats.nr_legs += 1
	
	if use_cache:
		dep_ast, dep_m, dep_t = mission[-1][:3]
		leg_key = (dep_ast, next_ast, dep_t, dep_m)
	
	if not use_cache or leg_key not in LEG_CACHE:
		rv_leg = rendezvous_leg(mission, next_ast, stats=stats, **kwargs)
		
		# add leg to cache. Will add a None value if leg is unfeasible
		if use_cache:
			LEG_CACHE[leg_key] = rv_leg
	else:
		# obtain the current leg's solution from the cache
		rv_leg = LEG_CACHE[leg_key]
	
	
	# if no feasible rendezvous leg could be found, `mission` is not extended
	if rv_leg is None:
		return False
	
	# extend `mission` with the rendezvous and self-flyby legs
	mission.append(rv_leg)
	fb_leg = self_flyby_leg(mission)
	mission.append(fb_leg)
	
	# if the mission's available mass or time is exhausted, remove the newly
	# added legs, and signal a failure to add and fully score the asteroid
	if final_mass(mission) < MASS_MIN or tof(mission) > TIME_MAX:
		mission[-2:] = []
		return False
	
	return True
	


# ==================================== ## ==================================== #
# ------------------------------------ # Define rendezvous and self-flyby legs

def rendezvous_leg(mission, next_ast, leg_dT=None, leg_dT_bounds=None,
	               obj_fun=None, stats=None, **kwargs):
	"""
	Define the leg that extends `mission` by performing a rendezvous with
	asteroid `next_ast`.
	"""
	if leg_dT is None and leg_dT_bounds is None:
		leg_dT_bounds = rvleg_dT_bounds
	if obj_fun is None:
		obj_fun = gtoc5_rendezvous
	
	dep_ast, dep_m, dep_t = mission[-1][:3]
	leg = lambert_optimize_dt(dep_ast, next_ast, dep_t, dep_m,
	                          leg_dT=leg_dT, leg_dT_bounds=leg_dT_bounds,
	                          obj_fun=obj_fun, mission=mission, stats=stats,
	                          **kwargs)
	if stats is not None:
		stats.nr_legs_distinct += 1
		stats.nr_legs_feasible += (1 if leg.feasible else 0)
	
	if not leg.feasible:
		return None
	
	mass_rv = leg.arr_m - MASS_EQUIPMENT
	
	return (
		next_ast,       # asteroid UID
		mass_rv,        # mass at asteroid, after the payload delivery
		leg.arr_t,      # Epoch at time of departure to self-flyby
		leg.dT,
		leg.dV
		)


def self_flyby_leg(mission):
	"""
	Define a self-flyby leg from/to the most recently visited asteroid in
	`mission`.
	"""
	ast, dep_m, dep_t = mission[-1][:3]
	
	mass_fb = dep_m * mass_fb_mult - MASS_PENETRATOR
	dT_fb = dep_m * dT_fb_mult
	
	return (
		ast,            # asteroid UID
		mass_fb,        # mass at asteroid, after the penetrator's delivery
		dep_t + dT_fb,  # Epoch at the end of the flyby
		dT_fb,
		dV_fb
		)
	


# ==================================== ## ==================================== #
# ------------------------------------ # Evaluation of rendezvous legs

class gtoc5_rendezvous(lambert_eval):
	
	def __init__(self, leg_dT, dep_ast, arr_ast, *args, **kwargs):
		# log asteroid ids; get their pk.planet instances
		assert type(dep_ast) is int, "Expected departure asteroid's ID."
		assert type(arr_ast) is int, "Expected arrival asteroid's ID."
		self.dep_ast_id = dep_ast
		self.arr_ast_id = arr_ast
		dep_ast = asteroids[dep_ast]
		arr_ast = asteroids[arr_ast]
		
		go_up = super(gtoc5_rendezvous, self)
		go_up.__init__(leg_dT, dep_ast, arr_ast, *args, **kwargs)
		
	
	def select(self, lamb_sol, v_body1, v_body2, stats=None, *args, **kwargs):
		"""
		Selects one of the Lambert's problem solutions
		(in case multiple revolution solutions were found).
		Selection criterion: solution with the smallest dV.
		"""
		if stats is not None:
			stats.nr_lambert += 1
		
		# get, per solution, the spacecraft's velocity at each body
		v1sc = lamb_sol.get_v1()
		v2sc = lamb_sol.get_v2()
		
		# determine each solution's dV
		solutions = []
		for v1, v2 in zip(v1sc, v2sc):
			dV1 = sqrt(sum((a - b) * (a - b) for (a, b) in zip(v_body1, v1)))
			dV2 = sqrt(sum((a - b) * (a - b) for (a, b) in zip(v_body2, v2)))
			
			if self.dep_ast_id == 0:
				# Earth departure
				# on the first leg, we get up to 5 km/s for free
				dV1 -= 5000.0
			else:
				# If we're not coming from Earth, we must take into account the
				# dV given by the self-flyby leg just performed at the departure
				# asteroid. That maneuver will deliver the projectile and leave
				# the spacecraft with a dV of 400 m/s relative to the asteroid,
				# in any direction we like.
				dV1 -= dV_fb_min
			
			solutions.append((dV1 + dV2, v1, v2))
		
		# pick the solution with smallest dV, and log the spacecraft's
		# velocities at each body
		self.dV, *self.v_sc = min(solutions)
		
	

class gtoc5_rendezvous_agg(gtoc5_rendezvous):
	
	def get_value(self):
		"Key value determining the instance's solution quality."
		# resource_rating should be maximized, so we return here instead its
		# negative for minimization (by lambert_optimize_dt, for instance).
		return - self.resource_rating
		
	
	def inspect(self, mission=None, *args, **kwargs):
		assert mission is not None, 'Unknown `mission`. ' \
			'Resource rating cannot be calculated'
		
		super(gtoc5_rendezvous_agg, self).inspect(*args, **kwargs)
		
		if self.feasible:
			# get aggregate rating, considering arrival mass after the current
			# leg, and mission's time of flight including this leg's dT
			self.resource_rating = rate_traj(self.arr_m, tof(mission) + self.dT)
		
	
	def fail(self):
		super(gtoc5_rendezvous_agg, self).fail()
		# The resource rating ranges in [0, 1]. We signal failure to optimize
		# leg with an out-of-bounds value below the worst possible rating (0.0).
		self.resource_rating = -1.
		
	
