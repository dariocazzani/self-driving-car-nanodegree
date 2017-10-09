port=${1:-8880}
docker run --privileged --rm -it \
    -v $PWD:/root/self-driving-car-nanodegree \
    -w /root/self-driving-car-nanodegree \
    -p $port:$port \
    carnd:term1 bash
