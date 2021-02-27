
from gemini.utils import *


class CodeNodeBase(object):

    __slots__ = [
        '_is_root',
        '_parent',
        '_src',
        '_src_file',
        '_ast',
        '_env',
        '_sub_code_nodes',
    ]

    def __init__(self, parent):
        # root node have no parent, if and only if
        self._parent = parent
        if parent is None:
            self._is_root = True
        else:
            self._is_root = False

        self._src = ""
        self._src_file = ""
        self._ast = ""
        self._env = ""
        self._sub_code_nodes = []

    @property
    def sub_code_nodes(self):
        return self._sub_code_nodes

    def add_code_node(self, value):
        assert isinstance(value, CodeNodeBase)
        assert value.parent is not None
        assert value.is_root is False
        self._sub_code_nodes.append(value)
        return

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
        if self._src == "" and self._ast == None:
            assert 0, 'CodeNode has no src and ast'
        elif self._src == "" and self._ast is not None:
            self._src = astunparse.unparse(self._ast)
        return self._src

    @src.setter
    def src(self, value):
        assert value is not None, 'received NoneType, not expected'
        assert isinstance(value, basestring), "expected <type 'basestring'>, got {}".format(type(value))
        self._src = value

    # getter and setter of src_file
    @property
    def src_file(self):
        return self._src_file

    @src_file.setter
    def src_file(self, value):
        assert value is not None, 'received NoneType, not expected'
        assert isinstance(value, basestring), "expected <type 'basestring'>, got {}".format(type(value))
        self._src_file = value

    # getter and setter of ast
    @property
    def ast(self):
        if self._ast is None and self._src == "":
            assert 0, 'CodeNode has no src and ast'
        elif self._ast is None and self._src != "":
            self._ast = ast.parse(self._src)
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

    def parse_modules(self):
        return

    def execute(self):
        return

