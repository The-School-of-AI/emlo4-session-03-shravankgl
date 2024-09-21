FROM shravankgl/elmo4-tsai:session2

WORKDIR /workspace

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY train.py check_train.py /workspace/

CMD ["python", "train.py"]