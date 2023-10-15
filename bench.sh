#!/bin/bash
source env.sh

RICHBENCH=src/rich-bench/richbench
python3 "$RICHBENCH" benchmarks $*
