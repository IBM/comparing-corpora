#!/bin/bash
#python setup.py sdist bdist_wheel upload -r local
echo Removing .env file from final distribution
echo Cleaning ./dist folder
rm  -f ./dist/*
echo Cleaning ./build folder
rm  -rf ./build
echo Running build
python3 setup.py sdist bdist_wheel
