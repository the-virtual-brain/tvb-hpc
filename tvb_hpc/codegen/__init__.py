"""
Handle code generation tasks. Likely to become a module with
task specific modules.

"""

from .base import BaseSpec, BaseCodeGen
from .cfun import CfunGen1
from .network import NetGen1
from .model import ModelGen1
