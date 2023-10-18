#!/bin/bash
source env.sh

LOG_FILE='pytest.log'
python3 -m pytest $* 4<&1 5<&2 1>&2>&>(tee >(sed -r 's/\x1B\[([0-9]{1,3}(;[0-9]{1,2};?)?)?[mGK]//g' > $LOG_FILE))
