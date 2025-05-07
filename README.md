# Business Card NLP

This project focuses on extracting and processing information from business cards using Natural Language Processing (NLP) techniques, leveraging libraries such as spaCy, Tesseract, and pytesseract for Named Entity Recognition (NER) and Optical Character Recognition (OCR).

## Project Overview

The goal is to extract key entities (e.g., name, email, phone number, organization) from business card images, preprocess the data, train a custom NER model using spaCy, and deploy the solution via a Flask-based web application.

---

## App Screenshot

![App Screenshot 1](https://github.com/Yhaziz/NERScan-Pro-Extraction-intelligente-de-donn-es-de-cartes-de-visite-par-IA/blob/master/doc.png?raw=true)
![App Screenshot 2](https://github.com/Yhaziz/NERScan-Pro-Extraction-intelligente-de-donn-es-de-cartes-de-visite-par-IA/blob/master/doc%20ia.png?raw=true)

---

## Setup Instructions

### Prerequisites
1. **Install Python**: Ensure Python (version 3.11.0) is installed on your system. Download from [python.org](https://www.python.org/downloads/release/python-3110/).
2. **Install Tesseract**: 
   - Download and install Tesseract OCR from [this link](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.4.0.20240606.exe).
   - Add the Tesseract executable path to your system's environment variables (e.g., `C:\Users\name\AppData\Local\Programs\Tesseract-OCR` on Windows).
3. **Install Docker**: Ensure Docker and Docker Compose are installed on your system. Download from [docker.com](https://www.docker.com/products/docker-desktop/).

### Project Setup
1. **Create a Virtual Environment**:
   ```bash
   python -m venv .env
   ```
2. **Activate the Virtual Environment**:
   - On Windows:
     ```bash
     .env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .env/bin/activate
     ```
3. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Install pytesseract**:
   ```bash
   pip install pytesseract
   ```
5. **Install spaCy**:
   ```bash
   pip install -U spacy
   ```
6. **Download spaCy English Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

7. **Run Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

---



## pytesseract Levels
pytesseract supports five levels for text segmentation:
1. **Image**: Treats the entire image as a single block.
2. **Block**: Divides the image into blocks of text.
3. **Paragraph**: Segments text into paragraphs.
4. **Line**: Extracts individual lines of text.
5. **Words**: Extracts individual words.

---

## Labeling
The project uses **BIO/IOB** (Beginning, Inside, Outside) tagging for entity annotation. Example:

| Token       | Label   |
|-------------|---------|
| John        | B-PER   |
| Doe         | I-PER   |
| @           | O       |
| example.com | O       |

### Entities
The model recognizes the following entities:
- **PER**: Person (Name)
- **ORG**: Organization
- **EMAIL**: Email Address
- **PHONE**: Phone Number
- **ADDRESS**: Physical Address

---

## Data Preprocessing
- **Word Positioning**: The position of each word on the business card is captured to aid in entity extraction.
- **Data Storage**:
  - Training and test datasets are saved as pickle files:
    ```python
    import pickle
    pickle.dump(TrainData, open('./data/TrainData.pickle', mode='wb'))
    pickle.dump(TestData, open('./data/TestData.pickle', mode='wb'))
    ```
  - **Explanation**:
    - `pickle.dump(obj, file)` serializes the object `obj` into a binary file.
    - `mode='wb'` indicates writing in binary mode.
    - Files `TrainData.pickle` and `TestData.pickle` store the training and test datasets, respectively, for later use.

---

## Training a Named Entity Recognition (NER) Model with spaCy

### 1. Prepare Configuration
- Download the `base_config.cfg` from [spaCy’s training page](https://spacy.io/usage/training).
- Save it in the project’s root directory.
- Generate the full configuration file:
  ```bash
  python -m spacy init fill-config ./base_config.cfg ./config.cfg
  ```

### 2. Prepare Data
- Use the provided `preprocess.py` script to convert data into spaCy’s binary format (`.spacy`).
- **preprocess.py** Explanation:
  ```python
  import spacy
  from spacy.tokens import DocBin
  import pickle

  nlp = spacy.blank("en")
  training_data = pickle.load(open('./data/TrainData.pickle', 'rb'))
  testing_data = pickle.load(open('./data/TestData.pickle', 'rb'))

  # Process training data
  db = DocBin()
  for text, annotations in training_data:
      doc = nlp(text)
      ents = []
      for start, end, label in annotations['entities']:
          span = doc.char_span(start, end, label=label)
          ents.append(span)
      doc.ents = ents
      db.add(doc)
  db.to_disk("./data/train.spacy")

  # Process test data
  db_test = DocBin()
  for text, annotations in testing_data:
      doc = nlp(text)
      ents = []
      for start, end, label in annotations['entities']:
          span = doc.char_span(start, end, label=label)
          ents.append(span)
      doc.ents = ents
      db_test.add(doc)
  db_test.to_disk("./data/test.spacy")
  ```
- Run the script:
  ```bash
  python preprocess.py
  ```
- This generates `train.spacy` and `test.spacy` files.

### 3. Train the Model
- Run the training command:
  ```bash
  python -m spacy train .\config.cfg --output .\output\ --paths.train .\data\train.spacy --paths.dev .\data\test.spacy
  ```
- **Output**:
  - Trained models are saved in `output/model-best` and `output/model-last`.
  - Training logs show metrics like F-score, precision, recall, and loss.

---

## Parser
The `parser(text, label)` function cleans and formats extracted text based on the entity type (e.g., email, phone, name). It ensures data consistency for downstream use.

### Predictions
- The `04_predictions.ipynb` notebook contains the prediction logic.
- Convert it to a `.py` file for use in the web app:
  ```bash
  jupyter nbconvert --to script 04_predictions.ipynb
  ```

### Improving Model Performance
- **Hyperparameter Tuning**: Adjust learning rate, batch size, or epochs in `config.cfg`.
- **Data Augmentation**: Add more diverse business card samples.
- **Fine-tuning**: Use a pre-trained spaCy model (e.g., `en_core_web_sm`) as a starting point.
- **Error Analysis**: Inspect misclassified entities to refine annotations.

---

## Web Application

### Setup
1. **Create a Virtual Environment**:
   ```bash
   python -m venv .envp
   ```
2. **Activate the Virtual Environment**:
   - On Windows:
     ```bash
     .envp\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .envp/bin/activate
     ```
3. **Generate `requirements_app.txt`**:
   - In the project directory with `requirements.txt`, run:
     ```bash
     pip freeze > requirements_app.txt
     ```
4. **Install Requirements**:
   ```bash
   pip install -r requirements_app.txt
   ```
5. **Install Flask**:
   ```bash
   pip install flask
   ```

### Implementation
- The web app uses **Flask** to render HTML templates.
- Create a `templates` folder in the project directory and add HTML files (e.g., `index.html`).
- Example Flask app structure:
  ```python
  from flask import Flask, render_template

  app = Flask(__name__)

  @app.route('/')
  def index():
      return render_template('index.html')

  if __name__ == '__main__':
      app.run(debug=True, host='0.0.0.0')
  ```
- **Templates**: Store HTML files in the `templates` folder and render them using `render_template`.

---



## Running the Web App with Docker


1. **Build the Docker Image**:
   - In the project directory, run:
     ```bash
     docker build -t business-card-nlp .
     ```

2. **Run the Docker Container**:
   - Start the container and map port 5000:
     ```bash
     docker run -p 5000:5000 business-card-nlp
     ```
   - Access the web app at `http://localhost:5000`.

### Using Docker Compose


1. **Build the Docker Compose Services**:
   - In the project directory, run:
     ```bash
     docker-compose build
     ```

2. **Run the Docker Compose Container**:
   - Start the container:
     ```bash
     docker-compose up
     ```
   - Access the web app at `http://localhost:5000`.

---
