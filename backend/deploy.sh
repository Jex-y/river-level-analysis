#!/bin/bash
TAG=europe-west2-docker.pkg.dev/durham-river-level/containers/backend:latest

docker pull $TAG || true
docker build -t $TAG --cache-from $TAG .
docker push $TAG
gcloud run deploy backend --image $TAG --region europe-west2 --platform managed --allow-unauthenticated
