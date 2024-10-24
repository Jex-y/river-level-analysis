# First check that model.onnx and config.json are in the upload dir
if [ ! -f "$1/model.onnx" ]; then
echo "model.onnx not found in $1"
    exit 1
fi

if [ ! -f "$1/config.json" ]; then
    echo "config.json not found in $1"
    exit 1
fi

name=$(basename $1)
path="gs://durham-river-level-models/$name"
gcloud storage cp -r $1 $path && echo "Deployed to $path" || (echo "Failed to upload" && exit 1)

# Used jq to write the model path to the config file

# new_model_path="$1/model.onnx"
# new_config_path="$1/config.json"
# # Keys are level_service.bucket_model_path and level_service.bucket_config_path
# content=$(jq --arg new_model_path "$new_model_path" --arg new_config_path "$new_config_path" '.level_service.bucket_model_path = $new_model_path | .level_service.bucket_config_path = $new_config_path' config.json)
# echo $content
