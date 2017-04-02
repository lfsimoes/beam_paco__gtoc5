# Multi-rendezvous Spacecraft Trajectory Optimization with Beam P-ACO #

This repository contains the code implementing the research described in the paper:

> Luís F. Simões, Dario Izzo, Evert Haasdijk, A. E. Eiben (2017) [Multi-rendezvous Spacecraft Trajectory Optimization with Beam P-ACO][RG] In: *Evolutionary Computation in Combinatorial Optimization: 17th European Conference, EvoCOP 2017 (Amsterdam, The Netherlands, April 19-21, 2017)* Edited by: Bin Hu, Manuel López-Ibáñez. 141-156 Springer. 


## Code Structure ##

...


## References ##

For additional information on the GTOC5 problem, or P-ACO, consult the references below.

See also the GTOC Portal [section for the GTOC5 problem][gtoc5@portal].

### GTOC5 / Spacecraft Trajectory Optimization ###

1. Grigoriev, I.S., Zapletin, M.P.: [GTOC5: Problem statement and notes on solution verification][ref1]. *Acta Futura* 8, 9–19 (2014)
2. Izzo, D., Simões, L.F., Yam, C.H., Biscani, F., Di Lorenzo, D., Addis, B., Cassioli, A.: [GTOC5: Results from the European Space Agency and University of Florence][ref2]. *Acta Futura* 8, 45–55 (2014)
3. Izzo, D., Hennes, D., Simões, L.F., Märtens, M.: [Designing complex interplanetary trajectories for the global trajectory optimization competitions][ref3]. In: Fasano, G., Pintér, J.D. (eds.) Space Engineering: Modeling and Optimization with Case Studies, pp. 151–176. Springer (2016)

### Population-based Ant Colony Optimization (P-ACO) ###

4. Guntsch, M., Middendorf, M.: [A Population Based Approach for ACO][ref4]. In: Cagnoni, S., Gottlieb, J., Hart, E., Middendorf, M., Raidl, G.R. (eds.) Applications of Evolutionary Computing: EvoWorkshops 2002: EvoCOP, EvoIASP, EvoSTIM/EvoPLAN. pp. 72–81. Springer, Berlin, Heidelberg (2002)
5. Guntsch, M., Middendorf, M.: [Applying Population Based ACO to Dynamic Optimization Problems][ref5]. In: Dorigo, M., Di Caro, G., Sampels, M. (eds.) Ant Algorithms: Third International Workshop, ANTS 2002. pp. 111–122. Springer, Berlin, Heidelberg (2002)
6. Guntsch, M., Middendorf, M.: [Solving Multi-criteria Optimization Problems with Population-Based ACO][ref6]. In: Fonseca, C.M., Fleming, P.J., Zitzler, E., Thiele, L., Deb, K. (eds.) Evolutionary Multi-Criterion Optimization: Second International Conference, EMO 2003. pp. 464–478. Springer, Berlin, Heidelberg (2003)
7. Guntsch, M.: [Ant algorithms in stochastic and multi-criteria environments][ref7]. Ph.D. thesis, Karlsruher Institut für Technologie (2004)
8. Oliveira, S., Hussin, M.S., Stützle, T., Roli, A., Dorigo, M.: [A Detailed Analysis of the Population-Based Ant Colony Optimization Algorithm for the TSP and the QAP][ref8]. Tech. Rep. TR/IRIDIA/2011-006, IRIDIA (Feb 2011) [[support data][ref8supp]]
9. Weise, T., Chiong, R., Lässig, J., Tang, K., Tsutsui, S., Chen, W., Michalewicz, Z., Yao, X.: [Benchmarking optimization algorithms: An open source framework for the traveling salesman problem][ref9]. IEEE Computational Intelligence Magazine 9(3), 40–52 (2014) [[GitHub][ref9code]]


## Dependencies ##

Below is the list of Python libraries on which the code depends.

#### Experiments (``gtoc5`` and ``paco`` modules, ``experiments*.py``): ####

* [PyKEP][pykep] 1.2.2 ([available here][pk122])
* numpy 1.10.4
* scipy 0.17.0
* tqdm 4.7.4

#### Experimental analysis: ####

* pandas 0.18.0
* matplotlib 1.5.1
* seaborn 0.7.1

The experiments reported in the paper were carried out in Python 3.4.4, using the above-listed versions of each library.




[RG]: https://www.researchgate.net/publication/315071181_Multi-rendezvous_Spacecraft_Trajectory_Optimization_with_Beam_P-ACO

[gtoc5@portal]: https://sophia.estec.esa.int/gtoc_portal/?page_id=25
[ref1]: http://dx.doi.org/10.2420/AF08.2014.9
[ref2]: http://dx.doi.org/10.2420/AF08.2014.45
[ref3]: http://dx.doi.org/10.1007/978-3-319-41508-6_6

[ref4]: http://dx.doi.org/10.1007/3-540-46004-7_8
[ref5]: http://dx.doi.org/10.1007/3-540-45724-0_10
[ref6]: http://dx.doi.org/10.1007/3-540-36970-8_33
[ref7]: http://d-nb.info/1013929756
[ref8]: http://iridia.ulb.ac.be/IridiaTrSeries/link/IridiaTr2011-006.pdf
[ref8supp]: http://iridia.ulb.ac.be/supp/IridiaSupp2011-010/
[ref9]: http://dx.doi.org/10.1109/MCI.2014.2326101
[ref9code]: https://github.com/optimizationBenchmarking/tspSuite

[pykep]: https://esa.github.io/pykep/
[pk122]: https://github.com/esa/pykep/releases/tag/1.2.2
