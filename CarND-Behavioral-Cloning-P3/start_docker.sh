docker run --privileged --rm -it \
    -v $PWD:/root/CarND-Behavioral-Cloning-P3 \
    -w /root/CarND-Behavioral-Cloning-P3 \
    -p 8888:8888 \
    -p 4567:4567 \
    carnd:term1 bash
