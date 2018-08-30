# TVB-HPC

This is a Python package for generating code for parameter sweeps and Bayesian
inversion.

## Quickstart

```
git clone https://gitlab.thevirtualbrain.org/tvb/hpc
python -m venv env
. env/bin/activate
pip install numpy
pip install -r requirements.txt
python -m unittest tvb_hpc.tests
```

## With Docker

```
docker pull maedoc/tvb-hpc
```

and run the tests
```
docker run --rm -it -v ./:/root/hpc python -m unittest tvb_hpc.tests
```
(on Windows, replace `./` by `%CD%`.)

Get hacking with the [`examples`](examples) or [tests](tvb_hpc/tests.py).

