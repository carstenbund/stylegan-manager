#!/bin/bash

sudo docker run -it --rm --runtime nvidia \
          -v `pwd`:/scratch --workdir /scratch \
          -e NVIDIA_VISIBLE_DEVICES=all \
          -e NVIDIA_DRIVER_CAPABILITIES=all \
          -e HOME=/scratch stylegan3  \
          python3 stylegan_server.py --outdir=images/${series}/${network} \
                  --trunc=1 --seeds=${start_amount}-${amount} \
                  --network=data/${series}/network-snapshot-${network}.pkl

