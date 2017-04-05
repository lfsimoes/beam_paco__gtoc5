# 
# Copyright (c) 2017 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the MIT License. See the LICENSE file for details.


from math import sqrt, sin, cos, acos, pi

import numpy as np
import PyKEP as pk

from .constants import MU_SUN, AU, EARTH_VELOCITY, DAY2SEC, asteroids



# ==================================== ## ==================================== #

def edelbaum_dv(ast1, ast2, t):
	"""
	Edelbaum's equation for constant acceleration, circle to inclined circle
	orbit transfer.
	
	Implements Eq. 45 from:
	Theodore N. Edelbaum. "Propulsion Requirements for Controllable Satellites",
	ARS Journal, Vol. 31, No. 8 (1961), pp. 1079-1089.
	http://dx.doi.org/10.2514/8.5723
	"""
	(a1, _, i1, W1, _, _) = ast1.osculating_elements(t)
	(a2, _, i2, W2, _, _) = ast2.osculating_elements(t)
	
	vc1 = sqrt(MU_SUN / a1)
	vc2 = sqrt(MU_SUN / a2)
	
	cos_i_rel = cos(i1) * cos(i2) + sin(i1) * sin(i2) * cos(W1) * cos(W2) + \
	            sin(i1) * sin(i2) * sin(W1) * sin(W2)
	if cos_i_rel > 1 or cos_i_rel < -1:
		cos_i_rel = 1
	i_rel = acos(cos_i_rel)
	
	dV = sqrt(vc1 * vc1 - 2. * vc1 * vc2 * cos(pi / 2. * i_rel) + vc2 * vc2)
	return dV


def rate__edelbaum(dep_ast, dep_t, **kwargs):
	dep_ast = asteroids[dep_ast]
	dep_t = pk.epoch(dep_t, 'mjd')
	return [edelbaum_dv(dep_ast, arr_ast, dep_t)
	        for arr_ast in asteroids]



# ==================================== ## ==================================== #

def eph_normalize(eph, refn_r=AU, refn_v=EARTH_VELOCITY):
	"""
	Normalize a body's ephemeris with respect to the given reference values:
	    (r / refn_r, v / refn_v)
	If unspecified, reference defaults to refn_r = AU, refn_v = EARTH_VELOCITY.
	Accepts either a single ephemeris, or a matrix with one per row.
	
	Example:
	>>> eph_normalize(pk.planet.gtoc5(1).eph(0))
	array([-3.41381819, -0.57375407, -0.28305754,
	        0.13551015, -0.43057761, -0.02779567])
	"""
	if type(eph) is tuple:
		eph = np.hstack(eph)
	e = eph.reshape(-1, 6)
	
	# normalize r
	e[:, :3] /= refn_r
	# normalize v
	e[:,-3:] /= refn_v
	
	return e.reshape(eph.shape)



def eph_reference(ref_ast, t):
	"""
	Calculate reference r and v magnitudes for use in the normalization
	of asteroid ephemeris.
	
	Returns:
		(|r|, |v|)
	where (r,v) is the ephemeris of the given asteroid `ref_ast` (id, or object)
	at epoch `t` (int/float, or epoch object).
	"""
	if type(ref_ast) is int:
		ref_ast = asteroids[ref_ast]
	if type(t) in [float, int]:
		t = pk.epoch(t, 'mjd')
	
	r, v = ref_ast.eph(t)
	return np.linalg.norm(r), np.linalg.norm(v)



def eph_matrix(asts, t, ref_ast=None):
	"""
	Given a list `asts` of asteroids (either by IDs, or as PyKEP.planet objects)
	and an epoch `t` (either as an mjd int/float, or as a PyKEP.epoch object),
	produce the matrix with their normalized ephemerides.
	
	If a reference asteroid `ref_ast` is given, its ephemeris will be used
	as reference for the normalization. Otherwise, AU, EARTH_VELOCITY will
	be used instead.
	"""
	if type(asts[0]) is int:
		asts = [asteroids[ast] for a in asts]
	if type(t) in [float, int]:
		t = pk.epoch(t, 'mjd')
	
	if ref_ast is None:
		norm_ref = AU, EARTH_VELOCITY
	else:
		norm_ref = eph_reference(ref_ast, t)
	
	# obtain all asteroids' ephemerides
	eph = np.array([a.eph(t) for a in asts])
	# reshape matrix, so each asteroid's ephemeris will be represented by
	# a single 6 dimensional vector
	eph = eph.reshape((-1, 6))
	
	# normalize the full matrix
	return eph_normalize(eph, *norm_ref)



def rate__euclidean(curr_ast, t, **kwargs):
	"""
	Euclidean Phasing Indicator.
	See: http://arxiv.org/pdf/1511.00821.pdf#page=12
	
	Estimates the cost of a transfer from departure asteroid `dep_ast` at epoch
	`dep_t`, towards each of the available asteroids.
	"""
	eph = eph_matrix(asteroids, t, asteroids[curr_ast])
	return np.linalg.norm(eph - eph[curr_ast], axis=1)



# ==================================== ## ==================================== #

def orbital_indicator(body, t, dT, neg_v=False, **kwargs):
	# reimplemented from:
	# https://github.com/esa/pykep/blob/master/PyKEP/phasing/_knn.py#L46
	(r1, r2, r3), (v1, v2, v3) = body.eph(t)
	r1 /= dT
	r2 /= dT
	r3 /= dT
	if neg_v:
		return (r1 - v1, r2 - v2, r3 - v3,
		        r1, r2, r3)
	else:
		return (r1 + v1, r2 + v2, r3 + v3,
		        r1, r2, r3)


def rate__orbital(dep_ast, dep_t, leg_dT, **kwargs):
	"""
	Orbital Phasing Indicator.
	See: http://arxiv.org/pdf/1511.00821.pdf#page=12
	
	Estimates the cost of a `leg_dT` days transfer from departure asteroid
	`dep_ast` at epoch `dep_t`, towards each of the available asteroids.
	"""
	dep_t = pk.epoch(dep_t, 'mjd')
	leg_dT *= DAY2SEC
	orbi = [orbital_indicator(ast, dep_t, leg_dT, **kwargs)
	        for ast in asteroids]
	ast_orbi = np.array(orbi[dep_ast])
	return np.linalg.norm(ast_orbi - orbi, axis=1)


def rate__orbital_2(dep_ast, dep_t, leg_dT, **kwargs):
	"""
	Refinement over the Orbital indicator.
	Attempts to correct bad decisions made by the linear model by checking
	which asteroids are close at arrival, and not only which are close at the
	beginning.
	"""
	r1 = rate__orbital(dep_ast, dep_t, leg_dT)
	r2 = rate__orbital(dep_ast, dep_t + leg_dT, leg_dT, neg_v=True)
#	return np.minimum(r1, r2)
	return np.mean([r1, r2], axis=0)

