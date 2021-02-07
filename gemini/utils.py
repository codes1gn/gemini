import os

__all__ = [
    'vlog'
]

def vlog(*args, **kwargs):
  if bool(os.getenv("DEBUG_MODE", 'False').lower() in ['true', '1']):
    # TODO fit with python2
    print(args)
    print(kwargs)
    # print(*args, **kwargs)
  else:
    pass

