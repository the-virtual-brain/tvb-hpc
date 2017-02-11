include_paths = [
    '-Iinclude', '-Iextern/gbench/include', '-Iextern/gtest/googletest/include'
]

def FlagsForFile( filename, **kwargs ):
  return {
    'flags': [ '-x', 'c++', '-std=c++11', '-Wall', '-Wextra'] + include_paths
  }
