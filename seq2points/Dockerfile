FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

WORKDIR /aidata-nilm

COPY . /aidata-nilm

COPY requirements_essential.txt /aidata-nilm/

RUN pip install --no-cache-dir -r requirements_essential.txt
