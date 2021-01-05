if [ "$#" -ne 2 ]; then
    echo "sh $0 [workspace path on host machine] [image tag]"
    exit 1
fi

docker run --rm --name EventExtractionMRC -it --gpus all --mount type=bind,source=$1,target=/EventExtractionMRC tensorflow-event-extraction-mrc:$2 bash
