#!/bin/bash
source env.sh

if [ $# -eq 0 ]; then
    args='html -d coverage'
else
    args=$*
fi

LOG_FILE='pytest.log'
python3 -m coverage run --include='src/*' -m pytest
python3 -m coverage $args
