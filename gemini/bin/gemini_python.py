import sys

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

    # construct config, use dummy string instead
    config = {'mode': 'sharding'}
    dump_to_file('dump_0_src.ast', compiler.dump())
    dump_to_file('dump_0_src.src', compiler.dump_src())
    compiler.apply_model_parallel(config)
    dump_to_file('dump_1_after.ast', compiler.dump())
    dump_to_file('dump_1_after.src', compiler.dump_src())

    use_ast = False
    # TODO(albert) have bug when not use_ast
    if not use_ast:
        try:
            compiler.compile_and_run(globals(), use_ast=False)
            print('try run src success')
        except Exception:
            print('try run src fail')
    else:
        try:
            compiler.compile_and_run(globals(), use_ast=True)
            print('try run ast success')
        except Exception:
            print('try run ast fail')


if __name__ == '__main__':
    main()
