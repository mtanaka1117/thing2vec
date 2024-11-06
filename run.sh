#!/bin/sh
docker image build -t thing2vec:test .
docker container run --rm --gpus "device=0" \
    -v $(pwd):/thing2vec \
    -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) \
    -it thing2vec:test bash