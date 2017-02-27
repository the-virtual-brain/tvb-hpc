# TVB-HPC

This is a Python package for generating code for parameter sweeps and Bayesian
inversion.

## Getting started

### Set up environment

Get the code and make a env to work in

```bash
git clone --recursive https://github.com/the-virtual-brain/tvb-hpc
PREFIX=$(pwd)/venv tvb-hpc/env/make-env.sh
source venv/activate
```

Be sure your git clone is recursive, otherwise some dependencies will not
be correctly obtained.

### Code

See our `examples` and test suite (`tvb_hpc/tests.py`) to get a
feel for what's implemented.
