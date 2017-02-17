# TVB-HPC

This is a Python package for generating code for parameter sweeps and Bayesian
inversion.

## Docs

Sphinx-built docs are [here](https://the-virtual-brain.github.io/tvb-hpc).

## Development

Current development is test-driven, run
```bash
TVB_LOG=DEBUG python3 -m unittest tvb_hpc.tests
```
Decrease `TVB_LOG` to `INFO` if it's too verbose. Please also check style with
```
flake8 tvb_hpc
```

If you're going to commit and push, consider using the pre commit hooks
```bash
ln -s $(pwd)/{.pre-commit,.git/hooks/pre-commit}
```
