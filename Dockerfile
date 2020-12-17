ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN pip3 install scaden

# Install tensorflow-gpu if GPU container
ARG GPU
RUN if [ "$GPU" = "GPU" ]; then \
    pip3 install tensorflow-gpu; \
    fi