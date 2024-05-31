#!/bin/bash

source env.sh

python3 app train -g 2 -p 2
code=$?

if [ $code -eq 0 ]; then
    echo -e '\nOK!'
else
    echo -e '\nFailure!'
fi;

exit $code
