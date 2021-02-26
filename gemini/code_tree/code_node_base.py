
from gemini.utils import *


class CodeNodeBase(object):

    __slots__ = [
        '_is_root',
        '_parent',
        '_src',
        '_ast',
        '_env',
    ]

    def __init__(self, parent):
        # root node have no parent, if and only if
        self._parent = parent
        if parent is None:
            self._is_root = True
        else:
            self._is_root = False

        self._src = ""
        self._ast = ""
        self._env = ""


    # getter and setter of is_root
    @property
    def is_root(self):
        return self._is_root

    @is_root.setter
    def is_root(self, value):
        assert isinstance(value, bool), 'expected bool value, but got {}'.format(type(value))
        self._is_root = value

    # getter and setter of parent
    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        assert value is not None, 'received NoneType, not expected'
        self._parent = value

    # getter and setter of src
    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, value):
        assert value is not None, 'received NoneType, not expected'
        assert isinstance(value, basestring), "expected <type 'basestring'>, got {}".format(type(value))
        self._src = value

    # getter and setter of ast
    @property
    def ast(self):
        return self._ast

    @ast.setter
    def ast(self, value):
        assert value is not None, 'received NoneType, not expected'
        assert isinstance(value, ast.AST), "expected <type 'ast.AST'>, got {}".format(type(value))
        self._ast = value

    # getter and setter of environment
    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, value):
        # add sanity check
        assert value is not None, 'received NoneType, not expected'
        assert isinstance(value, dict), "expected <type 'dict'>, got {}".format(type(value))
        self._env = value


    def execute_module(self, module_name):
        print('dummy execute_import on module {}'.format(module_name))
        return

    def execute(self):
        print('dummy execute')
        return

