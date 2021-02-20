import ast
import astunparse
import copy 

from gemini.utils import *

__all__ = [
    'MatmulShardingPass',
    'ShardingLeastDimTransformer',
    'ShardingLeastDimPostTransformer',
]

class BasePass:

    def __init__(self):
        pass

class MatmulShardingPass:

    def __init__(self):
        pass

class ShardingLeastDimTransformer(ast.NodeTransformer):
    
    __slots__ = [
        '_sharding_size',
        '_split_weights',
    ]

    def __init__(self, sharding_size=1):
        self._sharding_size = sharding_size
        self._split_weights = {'left': [], 'right': []}
        super(ast.NodeTransformer, self).__init__()

    @property
    def sharding_size(self):
        return self._sharding_size

    @property
    def split_weights(self):
        # type: (None) -> dict
        return self._split_weights

    def visit_BinOp(self, node):
        # print(astunparse.dump(node))
        # sanity check, do shortcut if not change
        if self._sharding_size == 1:
            return node

        # situation one, tf.matmul(a, b) + c
        if isinstance(node.left, ast.Call) and hasattr(
                node.left.func, 'attr') and (node.left.func.attr == "matmul"):
            # print('visiting ' + node.left.func.attr)
            lhs_id = node.left.args[0].id
            rhs_id = node.left.args[1].id

            # handle split weights, add weights id to the list
            self._split_weights['right'].append(rhs_id)
            self._split_weights['left'].append(lhs_id)
            # print('lhs id = ' + lhs_id)
            # print('rhs id = ' + rhs_id)
            func_attr = ast.Attribute(
                value=ast.Name(
                    id='tf',
                    ctx=ast.Load()
                ),
                attr='matmul',
                ctx=ast.Load()
            )
            reduce_attr = ast.Attribute(
                value=ast.Name(
                    id='tf',
                    ctx=ast.Load()
                ),
                attr='add_n',
                ctx=ast.Load()
            )
            _tmp = []
            for i in range(self._sharding_size):
                lhs_op = ast.Name(id=lhs_id + '_{}'.format(i), ctx=ast.Load())
                rhs_op = ast.Name(id=rhs_id + '_{}'.format(i), ctx=ast.Load())
                _tmp.append(ast.Call(
                    func=func_attr,
                    args=[
                        lhs_op,
                        rhs_op
                    ],
                    keywords=[],
                    starargs=None,
                    kwargs=None
                ))
            ret_node = ast.Call(
                func=reduce_attr,
                args=[ast.List(elts=_tmp, ctx=ast.Load())],
                keywords=[],
                starargs=None,
                kwargs=None
            )
            # ast_analysis(ret_node)
            node.left = ret_node

        ast.fix_missing_locations(node)
        return node


class ShardingLeastDimPostTransformer(ast.NodeVisitor):
    
    __slots__ = [
        '_sharding_size',
        '_split_weights',
    ]

    def __init__(self, tfr):
        assert isinstance(tfr, ShardingLeastDimTransformer),\
            "call __init__ of ShardingLeastDimPostTransformer; transformer is not of type ShardingLeastDimTransformer"
        self._sharding_size = tfr.sharding_size
        self._split_weights = tfr.split_weights
        super(ast.NodeVisitor, self).__init__()

    @property
    def sharding_size(self):
        return self._sharding_size

    @property
    def split_weights(self):
        return self._split_weights

    def visit_Assign(self, node):
        # vlog(astunparse.dump(node))
        # vlog(astunparse.dump(node.targets[0]))
        assert hasattr(node, 'gemini_parent'), "split_weights not have parents"
        parent_node = node.gemini_parent
        if hasattr(node.targets[0], 'id') and \
                node.targets[0].id in self._split_weights['left']:
            # print('before')
            # print(astunparse.dump(parent_node))
            # print('-----------------------\n')
            # add splitop after node
            self.insert_split_op(node, parent_node, split_axis=-1)
            # print('after')
            # print(astunparse.dump(parent_node))
            # print('-----------------------\n')

        if hasattr(node.targets[0], 'id') and \
                node.targets[0].id in self._split_weights['right']:
            # print('before')
            # print(astunparse.dump(parent_node))
            # print('-----------------------\n')
            # TODO check if weights are 2 dims, only handles matmul 2d
            self.split_dim_0_2d(node)
            # add nodes copys to parent node
            self.replicate_nodes(node, parent_node)
            # print('after')
            # print(astunparse.dump(parent_node))
            # print('-----------------------\n')

        self.generic_visit(node)

    # this method add split op after node, taking node id as input
    # split on axis=split_axis, default split on last dim

    def insert_split_op(self, node, parent_node, split_axis=-1):

        func_attr = ast.Attribute(
            value=ast.Name(id='tf', ctx=ast.Load()),
            attr='split',
            ctx=ast.Load()
        )

        func_args = []
        func_args.append(ast.Name(id=node.targets[0].id, ctx=ast.Load()))

        func_keywds = []
        func_keywds.append(
            ast.keyword(
                arg='num_or_size_splits',
                value=ast.Num(n=self.sharding_size)
            )
        )
        # bet the reduce dim is the last dim
        func_keywds.append(
            ast.keyword(
                arg='axis',
                value=ast.Num(
                    n=split_axis)))

        node_value = ast.Call(
            func=func_attr,
            args=func_args,
            keywords=func_keywds,
            starargs=None,
            kwargs=None
        )

        tuple_elts = []
        for idx in range(self.sharding_size):
            tuple_elts.append(
                ast.Name(
                    id=node.targets[0].id + '_{}'.format(idx),
                    ctx=ast.Store()
                )
            )
        node_targets = [ast.Tuple(elts=tuple_elts)]

        new_node = ast.Assign(targets=node_targets, value=node_value)
        old_index = parent_node.body.index(node)
        parent_node.body.insert(old_index + 1, new_node)
        setattr(new_node, 'gemini_parent', parent_node)
        ast.fix_missing_locations(parent_node)

    def split_dim_0_2d(self, node):
        # TODO check if weights are 2 dims, only handles matmul 2d
        assert len(
            node.value.args[0].elts) == 2, "yet, support matmul 2d only"
        node.value.args[0].elts[0] = ast.BinOp(
            left=node.value.args[0].elts[0],
            # TODO check if reduce dim is times of sharding size
            op=ast.FloorDiv(),
            right=ast.Num(n=self._sharding_size)
        )

    def replicate_nodes(self, node, parent_node):
        # add nodes copys to parent node
        for idx in range(self.sharding_size):
            if idx == 0:
                continue
            new_node = copy.deepcopy(node)
            new_node.targets[0].id += '_{}'.format(idx)
            old_index = parent_node.body.index(node)
            parent_node.body.insert(old_index + 1, new_node)
            setattr(new_node, 'gemini_parent', parent_node)

        # change id of var to var0 for first replica.
        # TODO(albert) may need to support Save Restore.
        node.targets[0].id += '_0'
        ast.fix_missing_locations(parent_node)