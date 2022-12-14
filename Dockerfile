FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
WORKDIR /code
ENV PORT=8000
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
COPY . /code

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
