#!/bin/bash

source env.sh

for i in sample/*; do
    echo -e "\nrun $i\n"
    python3 $i
    code=$?
    if [ $code -ne 0 ]; then
        break
    fi
done;

if [ $code -eq 0 ]; then
    echo -e '\nOK!'
else
    echo -e '\nFailure!'
fi;

exit $code
