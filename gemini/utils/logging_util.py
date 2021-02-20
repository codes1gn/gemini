import os
import ast
import astunparse

__all__ = [
    'vlog',
    'ast_analysis'
]

def ast_analysis(ast_node):
    print(astunparse.dump(ast_node))
    print(astunparse.unparse(ast_node))
    assert 0, "stop at ast_analysis"


def vlog(*args, **kwargs):
    if bool(os.getenv("DEBUG_MODE", 'False').lower() in ['true', '1']):
        # TODO fit with python2
        print(args)
        print(kwargs)
        # print(*args, **kwargs)
    else:
        pass
