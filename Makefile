init:
	pip install -r requirements.txt

lint:
	./scripts/pep8_linter.sh

install:
	mkdir -p build && mkdir -p build/install_tmp && python setup.py install --record build/install_tmp/dependencies.txt

uninstall:
	xargs rm -rf < build/install_tmp/dependencies.txt

tests:
	./scripts/run_unittest.sh

sample:
	./scripts/run_dump_ast_sample.sh

mnist:
	./scripts/run_mnist.sh

.PHONY: init install uninstall tests lint sample mnist

