#!/usr/bin/bash
# This script creates the environment required to run the code

rm -rf .venv
python3 -m virtualenv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 setup.py develop
