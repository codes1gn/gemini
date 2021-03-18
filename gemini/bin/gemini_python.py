import sys
import traceback
import copy
import importlib

from gemini.gemini_compiler import *
from gemini.utils import *


def main(argv=sys.argv[1:]):
    print('gemini compiler entry point')
    filename = copy.deepcopy(argv[0])
    arguments = copy.deepcopy(argv[1:])

    compiler = GeminiCompiler()
    src_code = read_src(filename)
    assert 'gemini_python' in sys.argv[0]

    # step 1, parse src code
    compiler.parse(src_code, filename=filename)
    compiler.dump(pretty=True, prefix='src_parse')
    assert 1, 'step 1 parse src'

    # patch, fix import gemini
    # _plugin = importlib.import_module('gemini.plugins.bert_plugin')
    # sys.modules['gemini'] = _plugin

    # step 2, parse modules
    compiler.parse_modules()
    compiler.dump(pretty=True, prefix='parse_module')
    assert 1, 'step 2 parse module'

    # TODO(albert) construct config, use dummy string instead
    config = Configuration()
    print(config)
    compiler.apply_model_parallel(config)
    compiler.dump(pretty=True, prefix='apply_{}_passes'.format(config.mode))
    assert 0, 'step 3 apply sharding mode'

    use_ast = False
    # TODO(albert) have bug when not use_ast
    if not use_ast:
        try:
            compiler.compile_and_run(use_ast=False)
            print('try run src success')
        except Exception:
            print('try run src fail')
            traceback.print_exc()
    else:
        try:
            compiler.compile_and_run(use_ast=True)
            print('try run ast success')
        except Exception:
            print('try run ast fail')
            traceback.print_exc()


if __name__ == '__main__':
    main()
