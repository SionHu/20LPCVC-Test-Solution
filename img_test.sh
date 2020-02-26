#!/bin/bash

FILES=~/Desktop/portrait/*.jpg
for file in $FILES
do
  python recognize_image.py -east east_detector.pb -i $file
done
