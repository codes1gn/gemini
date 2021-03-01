import unittest
import ast
import astunparse
import time

from gemini.code_tree import *
from gemini.utils import *


class TestGeminiCompilerDump(unittest.TestCase):

    def setUp(self):
        # prepare codes, compilers
        #   root
        #   /  \
        #  /    \
        # leaf1  leaf2
        #        /  \
        #       /    \
        #     leaf3 leaf4
        self.root_node = CodeNodeRoot()
        self.leaf_node1 = CodeNodeLeaf(self.root_node)
        self.leaf_node2 = CodeNodeLeaf(self.root_node)
        self.leaf_node3 = CodeNodeLeaf(self.leaf_node2)
        self.leaf_node4 = CodeNodeLeaf(self.leaf_node2)

    def tearDown(self):
        del self.root_node
        del self.leaf_node1
        del self.leaf_node2
        del self.leaf_node3
        del self.leaf_node4

    # method to test dump raw strings
    def test_parent_nodes(self):
        self.assertEqual(self.leaf_node1.parent, self.root_node)
        self.assertEqual(self.leaf_node2.parent, self.root_node)
        self.assertEqual(self.leaf_node3.parent, self.leaf_node2)
        self.assertEqual(self.leaf_node4.parent, self.leaf_node2)


if __name__ == '__main__':
    unittest.main()
