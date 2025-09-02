#!/bin/bash

series=000500
series=000084
network=cdf
network=demo
resolution=512
resolution=1024

outdir=images/${network}/${resolution}/${series} \
network_pkl=data/${network}/${resolution}/network-snapshot-${series}.pkl

sudo docker run -it --rm --runtime nvidia \
          -v `pwd`:/scratch --workdir /scratch \
	  -p 5000:5000 \
          -e NVIDIA_VISIBLE_DEVICES=all \
          -e NVIDIA_DRIVER_CAPABILITIES=all \
	  -e NETWORK_PKL=${network_pkl} \
          -e HOME=/scratch stylegan_server  \
          python3 stylegan_server.py --outdir=${outdir} \
                  --trunc=1 --seeds=${start_amount}-${amount} \
                  --network_pkl=${network_pkl}

