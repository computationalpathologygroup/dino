FROM python:3.9

RUN pip3 install --upgrade pip

# Copy requirements and install dependencies
COPY ./requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt


RUN mkdir /opt/ml
COPY ./code/ /opt/ml/code
RUN ls -la /opt/ml/code/*