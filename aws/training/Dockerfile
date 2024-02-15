FROM python:3.9

# Copy requirements and install dependencies
COPY ./requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt


RUN mkdir /opt
COPY ./dino/ /opt/training
RUN ls -la /opt/training/*

# Set the PYTHONPATH environment variable
ENV PYTHONPATH "${PYTHONPATH}:/code"

