init:
	pip install -r requirements.txt

install:
	mkdir -p build && mkdir -p build/install_tmp && python setup.py install --record build/install_tmp/dependencies.txt

uninstall:
	xargs rm -rf < build/install_tmp/dependencies.txt

test:
	py.test tests

.PHONY: init install uninstall
