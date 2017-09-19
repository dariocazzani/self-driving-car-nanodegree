docker run --privileged --rm -it \
    -v $PWD:/root/CarND-Advanced-Lane-Lines \
    -w /root/CarND-Advanced-Lane-Lines \
    -p 8888:8888 \
    carnd:term1 bash
