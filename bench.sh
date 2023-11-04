#!/bin/bash
source env.sh

python3 -m pytest benchmarks $*
