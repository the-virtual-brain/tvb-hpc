import os
import setuptools

version = os.environ.get('TVB_HPC_VER', 'git')

setuptools.setup(
    name='tvb-hpc',
    version=version,
    description='HPC code generation for TVB',
    author='TVB-HPC Contributors',
    url='https://gitlab.thevirtualbrain.org/tvb/hpc',
    packages=setuptools.find_packages(),
)
