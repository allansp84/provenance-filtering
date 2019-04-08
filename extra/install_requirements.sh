#!/bin/bash

for req in $(cat requirements.txt); do
    pip3 --no-cache-dir install $req;
done