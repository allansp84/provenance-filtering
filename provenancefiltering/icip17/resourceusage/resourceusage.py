# -*- coding: utf-8 -*-

# -- Adapted from
# Python Cookbook, 3rd Edition: Recipes for Mastering Python 3
# By David Beazley, Brian K. Jones. Publisher: O'Reilly Media

import sys
import signal
import resource


KB = 1024
MB = 1024 * KB
GB = 1024 * MB


def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))


def time_exceeded(signo, frame):
    print("Time's up!", signo, frame)
    raise SystemExit(1)


def set_max_runtime(seconds):
    # Install the signal handler and set a resource limit
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, hard))
    signal.signal(signal.SIGXCPU, time_exceeded)


def memory_usage_resource():
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        # ... it seems that in OSX the output is different units ...
        rusage_denom = rusage_denom * rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem


if __name__ == '__main__':
    set_max_runtime(15)
    while True:
        pass
