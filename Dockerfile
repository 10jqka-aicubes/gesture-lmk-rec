FROM 10jqkaaicubes/cuda:10.0-py3.7.9

COPY ./ /home/jovyan/gesture-lmk-rec

RUN cd /home/jovyan/gesture-lmk-rec  && \
    python -m pip install -r requirements.txt 