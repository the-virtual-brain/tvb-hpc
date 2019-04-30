# TVB-HPC

This is a Python package for generating code for parameter sweeps and Bayesian
inversion.

## Quickstart

```
git clone https://gitlab.thevirtualbrain.org/tvb/hpc tvb-hpc
cd tvb-hpc
python -m venv env
. env/bin/activate
pip install -r requirements.txt
python -m unittest tvb_hpc.tests
```

Get hacking with the [`examples`](examples) or [tests](tvb_hpc/tests.py).

## Targets

By default the requirements.txt file will have Numba installed, which
is an easy target to use, but other targets are straightforward to use

- `C` - ensure GCC or Clang are installed
- `OpenCL` - `pip install pyopencl`
