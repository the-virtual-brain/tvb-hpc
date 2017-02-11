import os

here = os.path.dirname(os.path.abspath(__file__))

here_include_paths = [
    'include', 'extern/gbench/include', 'extern/gtest/googletest/include'
]

include_paths = []
include_paths += ['-I' + os.path.join(here, hip) for hip in here_include_paths]

def FlagsForFile( filename, **kwargs ):
  return {
    'flags': [ '-x', 'c++', '-std=c++11', '-Wall', '-Wextra'] + include_paths
  }
