docker run --privileged --rm -it \
    -v $PWD:/root/self-driving-car-nanodegree \
    -w /root/self-driving-car-nanodegree \
    -p 8888:8888 \
    carnd:term1 bash
