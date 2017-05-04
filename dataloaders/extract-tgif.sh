#!/bin/bash
# extract things
mkdir tgif/jpgs
for f in tgif/gifs/*.gif
do
    subf="$(cut -d "/" -f3 <<< $f)"
    subf="$(cut -d. -f1 <<< $subf)"
    mkdir tgif/jpgs/$subf
    convert $f -resize "256^>" -coalesce tgif/jpgs/$subf/out.jpg

    identify -format "%T\n" $f > tgif/jpgs/$subf/duration.txt
done
