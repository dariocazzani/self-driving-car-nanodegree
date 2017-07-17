docker run --privileged --rm -it \
    -v $PWD:/root/CarND-LaneLines-P1 \
    -w /root/CarND-LaneLines-P1 \
    -p 8888:8888 \
    carnd:term1 bash
