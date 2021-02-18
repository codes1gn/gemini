#!/bin/sh

path=`dirname $0`'/..'
rpath=`realpath $path`

autopep8 --in-place --aggressive --recursive -vvvv $rpath/gemini/
autopep8 --in-place --aggressive --recursive -vvvv $rpath/tests/
autopep8 --in-place --aggressive --recursive -vvvv $rpath/samples/

