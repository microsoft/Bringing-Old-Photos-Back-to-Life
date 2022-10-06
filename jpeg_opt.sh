#!/bin/bash
shopt -s nullglob
shopt -s nocaseglob

for f in *.jpg; do
    convert "$f" -resize '4000x4000>' "opt/$f"
done