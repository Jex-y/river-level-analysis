
echo "Generating favicon.ico"
convert $1 -density 256x256 -define icon:auto-resize -colors 256 public/favicon.ico

echo "Generating favicon.svg"
cp $1 public/favicon.svg

sizes=(16 32 64 96 128 192 256 512)
for size in ${sizes[@]}; do
  echo "Generating icon-${size}.png"
  convert $1 -resize ${size}x${size} public/icon-${size}.png
done

echo "Generating apple-touch-icon.png"
convert $1 -resize 180x180 public/apple-touch-icon.png
