# 
# Copyright (c) 2017 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the MIT License. See the LICENSE file for details.


import numpy as np

from .constants import MASS_MAX, MASS_MIN, TIME_MAX



# ==================================== ## ==================================== #
# ------------------------------------ # Criteria aggregation

def rate_traj(m_f, tof, m_i=MASS_MAX, m_f_min=MASS_MIN, tof_max=TIME_MAX,
              ret_criteria=False, **kwargs):
	"""
	Calculate the "resource savings rating" (a.k.a.: "soft min" aggregation).
	Determines the extent to which the mass and time budgets available for the
	mission have been depleted by a trajectory that has `m_f` total mass left,
	and a cumulative time of flight `tof`.
	
	Returns a value in [0,1] for feasible trajectories:
	* 1.0 produced at the start of the mission, when `(m_f, tof) == (m_i, 0)`;
	* 0.0 reached when all usable mass has been depleted (`m_f == m_f_min`) or
	  the maximum time of flight has been reached (`tof == tof_max`).
	
	Returns negative values (unbounded) for unfeasible trajectories
	(`m_f < m_f_min` or `tof > tof_max`). In such cases, aggregation behaves
	as a regular "min" operator, over the normalized criteria values,
	with the worst performing criterion fully determining the returned value.
	
	For more information, see:
	* http://dx.doi.org/10.2420/AF08.2014.45
	* http://arxiv.org/abs/1511.00821
	
	Examples
	--------
	>>> rate_traj(m_f=1400., tof=2*365.25)
	0.3966101694915254
	"""
	# fraction of usable mass left
	m = (m_f - m_f_min) / (m_i - m_f_min)
	# fraction of mission time left
	t = (tof_max - tof) / tof_max
	
	# if trajectory is unfeasible in at least one of the objectives
	# return the maximum degree of (normalized) violation.
	if m < 0 or t < 0:
		if ret_criteria:
			return min(m, t), m, t
		else:
			return min(m, t)
	
	mt_sum = m + t
	if mt_sum == 0.0:
		# both resources are fully depleted.
		# return 0.0 and avoid the division by 0 below
		return 0.0 if not ret_criteria else (0.0, 0.0, 0.0)
	
	# Adaptive weights scheme:
	# a solution's criterion is given the more importance the least it's being
	# satisfied, thus focusing attention on it, and penalizing the overall
	# rating (assumes all objectives are to be maximized, and vary in [0,1]).
	wM = 1. - m / mt_sum
	wT = 1. - t / mt_sum
	agg = wM * m + wT * t
	
	if ret_criteria:
		# in addition to the aggregate rating, return also the criteria being
		# aggregated (the fractions of usable mass and time remaining)
		return agg, m, t
	else:
		return agg
	


def show__rate_traj(m_i=MASS_MAX, m_f_min=MASS_MIN, tof_max=TIME_MAX):
	"""
	Visualization of the aggregation function implemented in `rate_traj()`.
	Plots both the function over its input space (mass in kg, time in days),
	and over the space of normalized criteria.
	"""
	import matplotlib.pyplot as plt
	plt.ion()
	
	rate_traj__vect = np.vectorize(rate_traj)
	
	line_args = dict(c='black', ls=':', linewidth=1.25)
	label_args = dict(fontweight='bold')
	
	mass_left = np.linspace(m_f_min, m_i, 100)
	time_used = np.linspace(0, tof_max, 100)
	
	m, t = np.meshgrid(mass_left, time_used)
	agg, mn, tn = rate_traj__vect(m, t, m_i, m_f_min, tof_max, True)
	
	plt.figure()
	plt.pcolor(m, t/365.25, agg, cmap=plt.cm.viridis)
	cb = plt.colorbar()
	ct = plt.contour(m, t/365.25, agg, 10, cmap=plt.cm.gray_r)
	plt.clabel(ct, fmt='%.1f')
	plt.plot([m_i, m_f_min], [0,tof_max/365.25], **line_args)
	plt.xlabel('Mass available (kg)', **label_args)
	plt.ylabel('Mission time used (years)', **label_args)
	cb.set_label('Aggregate rating ("resource savings")', **label_args)
	
	plt.figure()
	plt.pcolor(mn, tn, agg, cmap=plt.cm.viridis)
	cb = plt.colorbar()
	ct = plt.contour(mn, tn, agg, 10, cmap=plt.cm.gray_r)
	plt.clabel(ct, fmt='%.1f')
	plt.plot([0,1], [0,1], **line_args)
	plt.xlabel('Mass available (fraction)', **label_args)
	plt.ylabel('Mission time available (fraction)', **label_args)
	cb.set_label('Aggregate rating ("resource savings")', **label_args)
	


# ==================================== ## ==================================== #
# ------------------------------------ # Pareto dominance

def pareto_front(points):
	"""
	Given a 2-dimensional sequence of `points`, obtain the list of indices into
	its Pareto front. Assumes both objectives are to be minimized.
	
	Implements, for two minimization objectives, the algorithm MAXIMA1/FILTER2
	described in [1] (Sec. 4.1.3). Complexity: "for d = 2, 3 the algorithm
	MAXIMA1 runs in optimal time \theta(N log N)" [1].
	
	[1]  Franco P. Preparata, Michael I. Shamos. Computational Geometry: An
	     Introduction. Springer-Verlag New York, Inc., New York, NY, 1985.
	[2]  H. T. Kung, F. Luccio, and F. P. Preparata. On Finding the Maxima
	     of a Set of Vectors. Journal of the ACM, 22(4):469-476, 1975.
	
	Examples
	--------
	>>> pts = [(5, 2), (1, 4), (1, 4), (9, 1), (3, 5), (4, 3), (7, 2), (1, 5)]
	>>> pareto_front(pts)
	[1, 5, 0, 3]
	>>> [pts[i] for i in pareto_front(pts)]
	[(1, 4), (4, 3), (5, 2), (9, 1)]
	"""
	if len(points) == 0:
		return []
	
	(a, b) = zip(*points)
	q = np.lexsort((b, a))
	# same as (but faster than):
	# q = sorted(list(range(len(points))), key=lambda i:tuple(points[i]))
	
	i = q[0]
	P = [i]
	ymin = points[i][1]
	for i in q[1:]:
		if points[i][1] < ymin:
			ymin = points[i][1]
			P.append(i)
	
	return P
	


def show__pareto_front(points, ax=None):
	"""
	Visualization of the Pareto front obtained by `pareto_front()` for the given
	sequence of `points`.
	
	Examples
	--------
	>>> show__pareto_front(np.random.randint(0, 20, size=(25, 2)))
	"""
	import matplotlib.pyplot as plt
	plt.ion()
	
	points = np.array(points)
	P = points[pareto_front(points)]
	
	if ax is None:
		ax = plt.figure().gca()
	ax.scatter(points[:,0], points[:,1])
	ax.step(P[:,0], P[:,1], c='r', where='post')
	
	return ax
	


def non_dominated_sorting(points):
	"""
	Nondominated sorting algorithm for two objectives.
	Assumes both objectives are to be minimized.
	
	Returns a list of lists, one per front, containing indices into `points`.
	These index the elements in `points` belonging to each front.
	
	Implements the algorithm described in Fig. 2 of:
	M. T. Jensen. Reducing the Run-time Complexity of Multi-Objective EAs: The
	NSGA-II and other Algorithms. IEEE Transactions on Evolutionary Computation,
	vol 7, no 5, pp 502-515. 2003.
	http://dx.doi.org/10.1109/TEVC.2003.817234
	
	Examples
	--------
	>>> pts = [(5, 2), (1, 4), (1, 4), (9, 1), (3, 5), (4, 3), (7, 2), (1, 5)]
	>>> nds = non_dominated_sorting(pts); nds
	[[1, 5, 0, 3], [2, 6], [7], [4]]
	>>> [[pts[i] for i in fr] for fr in nds]
	[[(1, 4), (4, 3), (5, 2), (9, 1)], [(1, 4), (7, 2)], [(1, 5)], [(3, 5)]]
	"""
	if len(points) == 0:
		return []
	
	p = points
	
	# sort the solutions to a sequence s_1, s_2, ..., s_N satisfying:
	# i < j => (x_1(s_i) < x_1(s_j)) or
	#          (x_1(s_i) = x_1(s_j) and x_2(s_i) =< x_2(s_j))
	(a, b) = zip(*p)
	q = np.lexsort((b, a))
	
	# e+1 == the current number of fronts
	e = 0
	
	F = [[q[0]]]
	
	# Invariant: F_1, ..., F_e hold a nondominated sorting of s_1, ..., s_(i-1)
	# The solutions in F_1, ..., F_e are not dominated by any of s_i, ..., S_N
	# (where N == len(q))
	for s_i in q[1:]:
		if p[s_i][1] < p[F[e][-1]][1]:
			# if s_i nondominated by F[e]
			for b in range(e + 1):
				# find lowest b such that s_i nondominated by F[b]
				if p[s_i][1] < p[F[b][-1]][1]:
					F[b].append(s_i)
					break
		else:
			# create a new front, and add s_i to it: F[e] == [s1]
			e += 1
			F.append([s_i])
	
	return F
	


def show__non_dominated_sorting(points, ax=None):
	"""
	Visualization of the non-dominated sorting of the given `points`, obtained
	by `non_dominated_sorting()`.
	
	Examples
	--------
	>>> show__non_dominated_sorting(np.random.randint(0, 20, size=(25, 2)))
	"""
	import matplotlib.pyplot as plt
	plt.ion()
	
	points = np.array(points)
	F = [points[fr] for fr in non_dominated_sorting(points)]
	
	if ax is None:
		ax = plt.figure().gca()
	
	ax.scatter(points[:,0], points[:,1])
	for f in F:
		ax.step(f[:,0], f[:,1], c='r', where='post')
	
	# plot the ideal, nadir and worst points
	# ref: http://www.iitk.ac.in/kangal/papers/k2008009.pdf
	pf = F[0]
	ideal = (pf[:,0].min(), pf[:,1].min())
	nadir = (pf[:,0].max(), pf[:,1].max())
	worst = (points[:,0].max(), points[:,1].max())
	
	args = dict(marker='*', lw=.75, s=100, zorder=3)
	ax.scatter(*ideal, label='ideal', c='green', **args)
	ax.scatter(*nadir, label='nadir', c='yellow', **args)
	ax.scatter(*worst, label='worst', c='red', **args)
	
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
	ax.legend(loc="upper left", scatterpoints=1, bbox_to_anchor=(1, 1))
	
	return ax
	


def hypervolume(front, reference_point):
	"""
	Compute the hypervolume that is dominated by a non-dominated front.
	Assumes a front composed of two objectives that are to be minimized.
	"""
	if len(front) == 0:
		return 0.0
	
	# obtain indices that lexicographically sort the front's objective values in
	# ascending order
	(obj_1, obj_2) = zip(*front)
	f = np.lexsort((obj_2, obj_1))
	
	v = 0
	r = reference_point
	for i in f:
		p = front[i]
		v += (r[0] - p[0]) * (r[1] - p[1])
		r = (r[0], p[1])
	
	return v
	

