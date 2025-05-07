#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import cv2
import pytesseract
import spacy
import re
import string

# Load NER model
model_ner = spacy.load('./output/model-best/')

def cleanText(txt):
    whitespace = string.whitespace
    punctuation = "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    tableWhitespace = str.maketrans('', '', whitespace)
    tablePunctuation = str.maketrans('', '', punctuation)
    text = str(txt)
    #text = text.lower()
    removewhitespace = text.translate(tableWhitespace)
    removepunctuation = removewhitespace.translate(tablePunctuation)
    return str(removepunctuation)

# Group the labels
class groupgen:
    def __init__(self):
        self.id = 0
        self.text = ''

    def getgroup(self, text):
        if self.text == text:
            return self.id
        else:
            self.id += 1
            self.text = text
            return self.id

# Parser
def parser(text, label):
    if label == 'PHONE':
        text = text.lower()
        text = re.sub(r'\D', '', text)
    elif label == 'EMAIL':
        text = text.lower()
        allow_special_char = '@_.\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char), '', text)
    elif label == 'WEB':
        text = text.lower()
        allow_special_char = ':/.%#\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char), '', text)
    elif label in ('NAME', 'DES'):
        text = text.lower()
        text = re.sub(r'[^a-z ]', '', text)
        text = text.title()
    elif label == 'ORG':
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]', '', text)
        text = text.title()
    return text

grp_gen = groupgen()

def getPredictions(image):
    # Validate input image
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image input. Expected a valid OpenCV image (numpy array).")

    # Extract data using Pytesseract
    tessData = pytesseract.image_to_data(image)
    if not tessData:
        raise ValueError("Pytesseract failed to extract data from the image.")

    # Convert into dataframe
    tessList = list(map(lambda x: x.split('\t'), tessData.split('\n')))
    df = pd.DataFrame(tessList[1:], columns=tessList[0])
    df.dropna(inplace=True)  # Drop missing values
    df['text'] = df['text'].apply(cleanText)

    # Convert data into content
    df_clean = df.query('text != ""').copy()  # Create explicit copy
    content = " ".join([w for w in df_clean['text']])

    # Get prediction from NER model
    doc = model_ner(content)
    
    # Convert doc to JSON
    docjson = doc.to_json()
    doc_text = docjson['text']

    # Creating tokens
    datafram_tokens = pd.DataFrame(docjson['tokens'])
    datafram_tokens['token'] = datafram_tokens[['start', 'end']].apply(
        lambda x: doc_text[x.iloc[0]:x.iloc[1]], axis=1)

    right_table = pd.DataFrame(docjson['ents'])[['start', 'label']]
    datafram_tokens = pd.merge(datafram_tokens, right_table, how='left', on='start')
    datafram_tokens.fillna('O', inplace=True)

    # Join label to df_clean dataframe
    df_clean.loc[:, 'end'] = df_clean['text'].apply(lambda x: len(x) + 1).cumsum() - 1
    df_clean.loc[:, 'start'] = df_clean[['text', 'end']].apply(
        lambda x: x['end'] - len(x['text']), axis=1)

    # Inner join with start
    dataframe_info = pd.merge(df_clean, datafram_tokens[['start', 'token', 'label']],
                             how='inner', on='start')

    # Bounding Box
    bb_df = dataframe_info.query("label != 'O'").copy()  # Create explicit copy
    bb_df.loc[:, 'label'] = bb_df['label'].apply(lambda x: x[2:])
    bb_df.loc[:, 'group'] = bb_df['label'].apply(grp_gen.getgroup)

    # Right and bottom of bounding box
    bb_df.loc[:, ['left', 'top', 'width', 'height']] = bb_df[['left', 'top', 'width', 'height']].astype(int)
    bb_df.loc[:, 'right'] = bb_df['left'] + bb_df['width']
    bb_df.loc[:, 'bottom'] = bb_df['top'] + bb_df['height']

    # Tagging: groupby group
    col_group = ['left', 'top', 'right', 'bottom', 'label', 'token', 'group']
    group_tag_img = bb_df[col_group].groupby(by='group')
    img_tagging = group_tag_img.agg({
        'left': 'min',
        'right': 'max',
        'top': 'min',
        'bottom': 'max',
        'label': np.unique,
        'token': lambda x: " ".join(x)
    })

    # Draw bounding boxes and labels
    img_bb = image.copy()
    for l, r, t, b, label, token in img_tagging.values:
        cv2.rectangle(img_bb, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(img_bb, str(label), (l, t), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    # Entities
    info_array = dataframe_info[['token', 'label']].values
    entities = dict(NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB=[])
    previous = 'O'

    for token, label in info_array:
        bio_tag = label[0]
        label_tag = label[2:]

        # Parse the token
        text = parser(token, label_tag)

        if bio_tag in ('B', 'I'):
            if previous != label_tag:
                entities[label_tag].append(text)
            else:
                if bio_tag == "B":
                    entities[label_tag].append(text)
                else:
                    if label_tag in ("NAME", 'ORG', 'DES'):
                        entities[label_tag][-1] = entities[label_tag][-1] + " " + text
                    else:
                        entities[label_tag][-1] = entities[label_tag][-1] + text

        previous = label_tag

    return img_bb, entities