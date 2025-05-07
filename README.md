# Business Card NLP

This project focuses on extracting and processing information from business cards using Natural Language Processing (NLP) techniques, leveraging libraries such as spaCy, Tesseract, and pytesseract for Named Entity Recognition (NER) and Optical Character Recognition (OCR).

## Project Overview

The goal is to extract key entities (e.g., name, email, phone number, organization) from business card images, preprocess the data, train a custom NER model using spaCy, and deploy the solution via a Flask-based web application.

---

## Setup Instructions

### Prerequisites
1. **Install Python**: Ensure Python (version 3.8 or higher) is installed on your system. Download from [python.org](https://www.python.org/downloads/).
2. **Install Tesseract**: 
   - Download and install Tesseract OCR from [this link](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-setup-3.05-dev.exe).
   - Add the Tesseract executable path to your system's environment variables (e.g., `C:\Program Files\Tesseract-OCR` on Windows).

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

---
