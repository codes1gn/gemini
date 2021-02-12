#!/bin/sh

path=`dirname $0`'/..'

autopep8 --in-place --aggressive --recursive -vvvv $path/gemini/
autopep8 --in-place --aggressive --recursive -vvvv $path/tests/
autopep8 --in-place --aggressive --recursive -vvvv $path/samples/

