# Stochastic-Shortest-Path

Maximizing the probability that the path length does not exceed a given threshold value (deadline).

Reimplementation of paper [[4]](https://github.com/Leot6/Stochastic-Shortest-Path#references), which is built on ref [[2]](https://github.com/Leot6/Stochastic-Shortest-Path#references) and ref [[3]](https://github.com/Leot6/Stochastic-Shortest-Path#references). Paper [[2]](https://github.com/Leot6/Stochastic-Shortest-Path#references) is built on ref [[1]](https://github.com/Leot6/Stochastic-Shortest-Path#references). The graph used is Manhattan, consisting of 4,092 nodes and 9,453 edges. Edges lengths are drawn from normal distributions.

The main function in `generate_graph.py` will generate the stochastic travel model using data from folder `graph`. A sample graph is ready to use in the root folder, named `NYC_NET.pickle`. `ssp.py` reimplement ref [[2]](https://github.com/Leot6/Stochastic-Shortest-Path#references), and `assp.py` reimplements ref [[4]](https://github.com/Leot6/Stochastic-Shortest-Path#references). `test.py` and file `NYC_REQ_DATA_20160501.pickle` is used to test how many taxi trips could find another path instead of the shortest mean path.

## References

1. Nikolova, E., Kelner, J.A., Brand, M. and Mitzenmacher, M., 2006, September. [Stochastic shortest paths via quasi-convex maximization](https://merl.com/publications/docs/TR2006-128.pdf). In European Symposium on Algorithms (pp. 552-563). Springer, Berlin, Heidelberg.
2. Lim, S., Balakrishnan, H., Gifford, D., Madden, S. and Rus, D., 2011. [Stochastic motion planning and applications to traffic](http://cocoa.lcs.mit.edu/papers/stoch-spaths.pdf). The International Journal of Robotics Research, 30(6), pp.699-712.
3. Nikolova, E., 2010. [Approximation algorithms for reliable stochastic combinatorial optimization](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.187.4510&rep=rep1&type=pdf). In Approximation, Randomization, and Combinatorial Optimization. Algorithms and Techniques (pp. 338-351). Springer, Berlin, Heidelberg.
4. Lim, S., Sommer, C., Nikolova, E. and Rus, D., 2013, July. [Practical route planning under delay uncertainty: Stochastic shortest path queries](http://roboticsproceedings.org/rss08/p32.pdf). In Robotics: Science and Systems (Vol. 8, No. 32, pp. 249-256).


