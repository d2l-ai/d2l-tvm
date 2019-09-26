#!/bin/bash

d2lbook build html

for fn in `find _build/html/_images/ -iname '*.svg' `; do
    if [[ $fn == *'qr_'* ]] ; then # || [[ $fn == *'output_'* ]]
        continue
    fi
    # rsvg-convert installed on ubuntu changes unit from px to pt, so evening no
    # change of the size makes the svg larger...
    rsvg-convert -z 1 -f svg -o tmp.svg $fn
    mv tmp.svg $fn
done
