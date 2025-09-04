#FROM dustynv/l4t-pytorch:r35.4.1

FROM nvcr.io/nvidia/pytorch:21.08-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAX_JOBS=8 

RUN pip install imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0
RUN pip install flask

WORKDIR /workspace

RUN (printf '#!/bin/bash\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
ENTRYPOINT ["/entry.sh"]

