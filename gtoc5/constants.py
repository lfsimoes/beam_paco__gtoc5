# 
# Copyright (c) 2017 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the MIT License. See the LICENSE file for details.


from math import sqrt, exp, radians

import numpy as np
import PyKEP as pk

from .ast_ephem import body, body_names



# ==================================== # Problem constants
# Source: GTOC5 problem statement
# http://dx.doi.org/10.2420/AF08.2014.9

# Sun's gravitational parameter µ_S, m^3/s^2
MU_SUN = 1.32712440018e11 * 1000**3

# Astronomical Unit AU, m
AU = 1.49597870691e8 * 1000

# Standard acceleration due to gravity, m/s^2
G0 = 9.80665

# Average Earth velocity, m/s
EARTH_VELOCITY = sqrt(MU_SUN / AU)

# Conversion factor from days to seconds
DAY2SEC = 86400.0
# Conversion factor from seconds to days
SEC2DAY = 1.0 / DAY2SEC
# Conversion factor from years to days
YEAR2DAY = 365.25
# Conversion factor from days to years
DAY2YEAR = 1.0 / YEAR2DAY


# Specific impulse of the spacecraft's engine, s
I_sp = 3000.0

# Spacecraft's maximum thrust level, N
T_max = 0.3


# "The spacecraft has a fixed initial mass, i.e. wet mass, m_i = 4000 kg"
MASS_MAX = 4000.0
# "The spacecraft dry mass is m_d >= 500 kg"
MASS_MIN = 500.0

# "The weight of such a scientific equipment is set to be 40 kg at each
# asteroid. The second asteroid encounter (fly-by) corresponds to the delivery
# of a 1 kg penetrator."
MASS_EQUIPMENT = 40.0
MASS_PENETRATOR = 1.0

# constraint for the flyby to deliver the penetrator:
# "flyby asteroid with a velocity not less than dV_min = 0.4 km/s."
dV_fb_min = 400.0	# m/s


# "The flight time, measured from start to the end must not exceed 15 years"
TIME_MAX = 15 * YEAR2DAY	# 5478.75 days

# "The year of launch must lie in the range 2015 to 2025, inclusive:
# 57023 MJD <= t_s <= 61041 MJD."
TRAJ_START_MIN = pk.epoch(57023, 'mjd')
TRAJ_START_MAX = pk.epoch(61041, 'mjd')
# >>> TRAJ_START_MIN, TRAJ_START_MAX
# (2015-Jan-01 00:00:00, 2026-Jan-01 00:00:00)

TRAJ_END_MIN = pk.epoch(TRAJ_START_MIN.mjd + TIME_MAX, 'mjd')
TRAJ_END_MAX = pk.epoch(TRAJ_START_MAX.mjd + TIME_MAX, 'mjd')
# >>> TRAJ_END_MIN, TRAJ_END_MAX
# (2029-Dec-31 18:00:00, 2040-Dec-31 18:00:00)



# ==================================== # GTOC5 asteroids (body objects)

# Earth's Keplerian orbital parameters
# Source: http://dx.doi.org/10.2420/AF08.2014.9 (Table 1)
_earth_op = (
	pk.epoch(54000, 'mjd'),     # t
	0.999988049532578 * AU,     # a
	1.67168116316e-2,           # e
#	radians(9.954353079654e-4), # i (value in the original problem statement)
	radians(8.854353079654e-4), # i
	radians(175.40647696473),   # W
	radians(287.61577546182),   # w
	radians(257.60683707535)    # M
	)
earth = pk.planet.keplerian(_earth_op[0], _earth_op[1:],
                            MU_SUN, 398601.19 * 1000**3, 0, 0, 'Earth')


# build `asteroids`, list of all body's `pk.planet.keplerian` objects
# https://esa.github.io/pykep/documentation/planets.html#PyKEP.planet.keplerian
_body = body.copy()
_body[:,1] *= AU
_body[:,3:] = np.deg2rad(_body[:,3:])

# (Earth placed at index 0; asteroids' ids then match their indices in the list)
asteroids = [earth] + [
	pk.planet.keplerian(pk.epoch(orbparam[0], 'mjd'), tuple(orbparam[1:]),
	                    MU_SUN, 0, 0, 0, name)
	for (orbparam, name) in zip(_body, body_names)
	]



# ==================================== # Model constants
# Source:
# http://dx.doi.org/10.2420/AF08.2014.45

# linear acceleration model for the self-flyby legs:

# dV of a self-flyby leg
dV_fb = dV_fb_min * (1 + sqrt(2))
# == 965.6854249492379 m/s

# multiplier for determining a self-flyby leg's dT
# (dT_fb = initial_mass * dT_fb_mult)
dT_fb_mult = dV_fb * SEC2DAY / T_max
# == 0.03725638213538727

# multiplier for determining the final mass following a self-flyby leg.
# Obtained from the Tsiolkovsky rocket equation, using the known dV_fb.
# (mass_fb = initial_mass * mass_fb_mult)
mass_fb_mult = exp(dV_fb / (-I_sp * G0))
# == 0.9677086973525064



# ==================================== # Search constants

# tolerance factor on a leg's maximum thrust.
# acceleration limited to at most 90% of the maximum supported by the spacecraft
thrust_tol = 0.9

# bounds (days) in which to search for a rendezvous leg's dT
#rvleg_dT_bounds = [60., 2. * YEAR2DAY]
rvleg_dT_bounds = [60., 500.]



# ==================================== #
