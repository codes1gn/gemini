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

sample_ast_dump:
	./scripts/run_dump_ast_sample.sh

sample_mnist:
	./scripts/run_mnist.sh

sample_bert:
	./scripts/run_bert.sh

clean:
	rm -rf ./build && rm -rf ./dist && rm -rf dump_ast && rm -rf dump_graph && rm -f log* && rm -rf *.egg-info

.PHONY: init install uninstall tests lint sample_ast_dump sample_mnist sample_bert

