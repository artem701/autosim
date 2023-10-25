#!/bin/bash

reqs=`cat requirements.txt <(pipreqs --print 2>/dev/null) | sort | uniq`
echo "$reqs" | tee requirements.txt
