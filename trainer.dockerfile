# Base image to build on top of:

FROM python:3.10-slim

#Next we are going to install some essentials in our image.
# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

#Lets copy over our application (the essential parts) from our computer to the container:
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir


#The entrypoint is the application that we want to run when the image is being executed:
#The u argument redirects any outputs (for instance prints) to out terminal. 
#If not included you would need to use docker logs to inspect your run.
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
