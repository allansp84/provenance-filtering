#!/bin/bash

for req in $(cat requirements.txt); do
    pip --no-cache-dir install $req;
done
