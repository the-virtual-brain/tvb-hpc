include_paths = [
    '-Iinclude',
    '-isystem', '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/include/python2.7',
    '-system', '/usr/local/Cellar/python3/3.6.0/Frameworks/Python.framework/Versions/3.6/include/python3.6m',
]

def FlagsForFile( filename, **kwargs ):
  return {
    'flags': [ '-x', 'c++', '-std=c++11', '-Wall', '-Wextra'] + include_paths
  }
