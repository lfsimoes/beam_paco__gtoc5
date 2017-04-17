"""
Compilation of high-scoring trajectories.
"""
# 
# Copyright (c) 2017 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the MIT License. See the LICENSE file for details.


from .gtoc5 import *



def build_traj(seq, **kwargs):
	mission = mission_to_1st_asteroid(seq[1])
	for ast in seq[2:]:
		if not add_asteroid(mission, ast, use_cache=False, **kwargs):
			break
	return mission



# ==================================== # Golden Path
# one of the best score 16 trajectories found by the ACT/GOL team during GTOC5

# replicating settings used in the competition's branch & prune tree search
# (chemical_B_and_P.py - find_min_dv())
_args = dict(grid=True, obj_fun=gtoc5_rendezvous,
             leg_dT_bounds=[100., 490.], nr_evals=40)

_seq = [0, 1712, 4893, 2579, 4813,  960, 5711, 4165, 5884,
           5174, 1059, 2891, 6008, 5264, 1899, 6834, 5311]

golden_path = build_traj(_seq, **_args)

assert score(golden_path) == 16 and \
       abs(resource_rating(golden_path) - 0.011875988961470869) < 1e-10, \
       'Failed validation: Golden Path'

# >>> score(golden_path), final_mass(golden_path), tof(golden_path) * DAY2YEAR
# (16.0, 550.7674621625735, 14.84919353516749)



# ==================================== # Spice
# the (single) score 17 trajectory found during the Beam P-ACO research
# https://arxiv.org/abs/1704.00702

_args = dict(grid=True, obj_fun=gtoc5_rendezvous,
             leg_dT_bounds=[60., 500.], nr_evals=50)

_seq = [0, 1712, 4893, 2579, 6979, 5469, 6740, 2445, 6301,
           5174, 5884, 4165, 4028, 6240, 3988, 1779, 6813, 3243]

spice = build_traj(_seq, **_args)

assert score(spice) == 17 and \
       abs(resource_rating(spice) - 0.0016809741417127071) < 1e-10, \
       'Failed validation: Spice'

# >>> score(spice), final_mass(spice), tof(spice) * DAY2YEAR
# (17.0, 503.1700581362939, 14.824982263414686)
