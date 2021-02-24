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

ast_dump_sample:
	./scripts/run_dump_ast_sample.sh

mnist_sample:
	./scripts/run_mnist.sh

bert_sample:
	./scripts/run_bert.sh

clean:
	rm -rf ./build && rm -rf ./dist && rm -rf dump_ast && rm -rf dump_graph && rm -f log* && rm -rf *.egg-info

.PHONY: init install uninstall tests lint ast_dump_sample mnist_sample bert_sample

