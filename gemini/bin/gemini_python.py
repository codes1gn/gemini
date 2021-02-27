import sys
import traceback

from gemini.gemini_compiler import *
from gemini.utils import *


def main(argv=sys.argv[1:]):
    print('gemini compiler entry point')
    filename = argv[0]
    arguments = argv[1:]

    compiler = GeminiCompiler()
    src_code = read_src(filename)

    # step 1, parse src code
    compiler.parse(src_code, filename=filename)
    dump_to_file('dump_0_parse_src.ast', compiler.dump())
    dump_to_file('dump_0_parse_src.src', compiler.dump_src())
    assert 1, 'after parse src'

    # step 2, parse modules
    compiler.parse_modules()
    dump_to_file('dump_1_parse_module.ast', compiler.dump())
    dump_to_file('dump_1_parse_module.src', compiler.dump_src())
    assert 0, 'after parse module'

    # TODO(albert) construct config, use dummy string instead
    config = {'mode': 'sharding'}
    compiler.apply_model_parallel(config)
    dump_to_file('dump_3_apply_{}_pass.ast'.format(config['mode']), compiler.dump())
    dump_to_file('dump_3_apply_{}_pass.src'.format(config['mode']), compiler.dump_src())

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
