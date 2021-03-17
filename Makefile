init:
	pip install -r requirements.txt
	git submodule init
	git submodule update

lint:
	./scripts/pep8_linter.sh

install:
	mkdir -p build && mkdir -p build/install_tmp && python setup.py install --record build/install_tmp/dependencies.txt

uninstall:
	xargs rm -rf < build/install_tmp/dependencies.txt

tests:
	./scripts/run_unittest.sh

communicate:
	./scripts/run_communicate.sh

samples/dump_ast:
	./scripts/run_dump_ast_sample.sh

samples/mnist:
	./scripts/run_mnist.sh

samples/mnist_with_python_vanilla:
	./scripts/run_mnist.sh vanilla

samples/bert:
	./scripts/run_bert.sh

# entry to run experimental bert code that tries to apply generic sharding mode by manually modify source code
samples/bert_experimental:
	./scripts/run_bert_experimental.sh

samples/imports:
	./scripts/run_imports.sh

samples/imports_with_python_vanilla:
	./scripts/run_imports.sh vanilla

clean:
	rm -rf ./build && rm -rf ./dist && rm -rf dump_ast && rm -rf dump_graph && rm -f log* && rm -rf *.egg-info

.PHONY: init install uninstall tests lint samples/dump_ast samples/mnist samples/bert samples/imports
