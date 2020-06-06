DOCKER="$1"
ACCESS="$2"

if [[ "$DOCKER" == "tf_cpu" ]]; then
    echo "Initialize ENV as tf_cpu"
    if [[ "$ACCESS" == "jupyter" ]]; then
        docker run -it --rm --name=tf_cpu -p 9999:8888 -v ${PWD}:/ws -w /ws tf_cpu jupyter notebook
    elif [[ "$ACCESS" == "cmd" ]]; then
        docker exec -it tf_cpu bash
    else
        echo "Bad ACCESS option"
    fi
elif [[ "$DOCKER" == "tf_gpu" ]]; then
    echo "Initialize ENV as tf_gpu"
    if [[ "$ACCESS" == "jupyter" ]]; then
        docker run -it --rm  --runtime=nvidia --name=tf_gpu -p 5555:8888 -v ${PWD}:/ws -w /ws tf_gpu jupyter notebook
    elif [[ "$ACCESS" == "cmd" ]]; then
        docker exec -it tf_gpu bash
    else
        echo "Bad ACCESS option"
    fi
elif [[ "$DOCKER" == "compy" ]]; then
    echo "Initialize ENV as compy"
    if [[ "$ACCESS" == "jupyter" ]]; then
        docker run -it --rm --name=compy -p 8888:8888 -v ${PWD}:/ws -w /ws compy jupyter notebook
    elif [[ "$ACCESS" == "cmd" ]]; then
        docker exec -it compy bash
    else
        echo "Bad ACCESS option"
    fi
else
    echo "No such docker container found."
fi
