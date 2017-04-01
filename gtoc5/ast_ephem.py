# 
# Copyright (c) 2017 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the MIT License. See the LICENSE file for details.


import os

import numpy as np



# ==================================== # Load asteroids' orbital parameters

_asts_file = os.path.dirname(os.path.abspath(__file__)) + '/ast_ephem.txt'
# ast_ephem.txt: reformatted version of the problem data file available at
# https://sophia.estec.esa.int/gtoc_portal/wp-content/uploads/2012/11/gtoc5_problem_data.txt


body = np.loadtxt(_asts_file, skiprows=3, usecols=range(7), delimiter='\t',
                  dtype=np.double)

# swapping the `w` and `Node` columns, so rows will
# follow PyKEP's convention: (a, e, i, W, w, M)
# https://esa.github.io/pykep/documentation/planets.html#PyKEP.planet.keplerian
body[:,[4,5]] = body[:,[5,4]]


body_names = np.loadtxt(_asts_file, skiprows=3, usecols=[7], delimiter='\t',
                        dtype=bytes).astype(np.str)
# bytes->str: http://stackoverflow.com/a/33656237
