# TVB-HPC

This is a Python package for generating code for parameter sweeps and Bayesian
inversion.

Get the Docker image
```
docker pull maedoc/tvb-hpc
```

and run the tests
```
docker run --rm -it -v ./:/root/hpc python -m unittest tvb_hpc.tests
```
(on Windows, replace `./` by `%CD%`.)

Get hacking with the [`examples`](examples) or [tests](tvb_hpc/tests.py).

## TODO

- ensure par sweeps etc built as domains
- rng
- make high level usage easier (tavg, bold, gain, fcd, etc)
- test on CUDA
- parallel numba
- simple SALib usage?
- chunking of state & vectorization?
- reach cuda hackathon performance numbers