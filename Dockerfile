FROM python:3.9-slim
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
COPY . /code

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT