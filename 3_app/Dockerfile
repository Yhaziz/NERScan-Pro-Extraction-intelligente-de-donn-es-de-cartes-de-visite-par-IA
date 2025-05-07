FROM python:3.11-slim


RUN apt-get update && apt-get install -y \
    libopencv-dev \
    tesseract-ocr \
    libtesseract-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements_app.txt .


RUN pip install --no-cache-dir -r requirements_app.txt


COPY main.py Predicitions.py settings.py utils.py ./
COPY static/ ./static/
COPY templates/ ./templates/
COPY output/ ./output/


EXPOSE 5000


CMD ["python", "main.py"]