FROM continuumio/miniconda3

RUN echo "creating working directory!!!"

WORKDIR /app

RUN echo "copying files from repo!!!"

COPY . /app

RUN echo "creating virtual env from yml file!!!"

RUN conda env create -f environment.yml

RUN echo "Activate the environment, and set entry point!!!"

SHELL ["conda", "run", "-n", "bias-detection", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "bias-detection"]

RUN echo "bias-detection env activated and image is ready!!!"


