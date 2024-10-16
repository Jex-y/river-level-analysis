#!/bin/bash

ICON_DIR=public
SIZES=(16 32 64 96 128 192 256 512)

echo "Generating icon.ico"
convert $1 -density 256x256 -define icon:auto-resize -colors 256 $ICON_DIR/icon.ico

echo "Generating icon.svg"
cp $1 $ICON_DIR/icon.svg

for size in ${SIZES[@]}; do
  echo "Generating icon-${size}.png"
  convert $1 -resize ${size}x${size} $ICON_DIR/icon-${size}.png
done

echo "Generating apple-icon.png"
convert $1 -resize 180x180 $ICON_DIR/apple-icon.png
