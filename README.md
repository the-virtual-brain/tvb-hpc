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

- fix hackathon example
- parallel numba
- simple SALib usage?