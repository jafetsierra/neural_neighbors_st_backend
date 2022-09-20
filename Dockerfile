FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install opencv-python
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY . /code

CMD uvicorn main:app --host 0.0.0.0 --port 8000
