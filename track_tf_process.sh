#!/bin/sh

PID=$( ps -A -o pid,cmd|grep "rainbow.py" | grep -v grep |head -n 1 | awk '{print $1}')
watch -n 1 tail /proc/$PID/fd/3
