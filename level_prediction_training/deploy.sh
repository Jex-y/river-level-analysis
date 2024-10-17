gcloud storage cp -r . gs://durham-river-level-models/$1 && echo "Deployed to gs://durjson-river-level-models/$1" || (echo "$1" && exit 1)
