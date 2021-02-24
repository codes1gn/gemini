import sys
import traceback

from gemini.gemini_compiler import *
from gemini.utils import *


def _get_compiler(filename, arguments):
    compiler = GeminiCompiler()
    src_code = read_src(filename)
    compiler.parse(src_code, filename=filename)
    return compiler


def main(argv=sys.argv[1:]):
    print('gemini compiler entry point')
    filename = argv[0]
    arguments = argv[1:]
    compiler = _get_compiler(filename, arguments)

    # TODO(albert) construct config, use dummy string instead
    config = {'mode': 'sharding'}
    dump_to_file('dump_0_before_{}_mode_pass.ast'.format(config['mode']), compiler.dump())
    dump_to_file('dump_0_before_{}_mode_pass.src'.format(config['mode']), compiler.dump_src())
    compiler.apply_model_parallel(config)
    dump_to_file('dump_1_after_{}_mode_pass.ast'.format(config['mode']), compiler.dump())
    dump_to_file('dump_1_after_{}_mode_pass.src'.format(config['mode']), compiler.dump_src())

    use_ast = False
    # TODO(albert) have bug when not use_ast
    if not use_ast:
        try:
            compiler.compile_and_run(globals(), use_ast=False)
            print('try run src success')
        except Exception:
            print('try run src fail')
            traceback.print_exc()
    else:
        try:
            compiler.compile_and_run(globals(), use_ast=True)
            print('try run ast success')
        except Exception:
            print('try run ast fail')
            traceback.print_exc()


if __name__ == '__main__':
    main()
