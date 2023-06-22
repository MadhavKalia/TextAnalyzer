#in terminal: python -m streamlit run main.py
#pip install PyPDF2
#pip install streamlit
#pip install pandas
#pip install cleantext
#pip install -U textblob
    #python -m textblob.downdload_corpora

import streamlit as sl
import pandas as pd
from textblob import TextBlob
import cleantext
import re
from PyPDF2 import PdfReader


sl.header("Text Analyzer")
with sl.expander("Analyze Text"):
    text = sl.text_input("Insert Text here:")
    if text:
        blob = TextBlob(text)
        sl.write("Polarity: ", round(blob.sentiment.polarity,2))
        sl.write("Subjectivity: ", round(blob.sentiment.subjectivity, 2))
        sl.write("Word Count:" , len(blob.words))
        sl.write("Character Count:" , len(blob))

    before = sl.text_input("Clean Text: ")
    if before:
        cleaned_text = cleantext.clean(
            before,
            clean_all=False,
            extra_spaces=True,
            stopwords=True,
            lowercase=True,
            numbers=True,
            punct=True
        )

        # Find removed words
        removed_words = re.findall(r"\b(\w+)\b", before)
        removed_words = set(removed_words) - set(cleaned_text.split())

        # Highlight removed words in red
        for word in removed_words:
            before = re.sub(rf"\b({word})\b", r"<span style='color:red'>\1</span>", before)

        # Display the modified text
        sl.markdown(before, unsafe_allow_html=True)

def score(input):
    blob2 = TextBlob(input)
    return blob2.sentiment.polarity

with sl.expander("Analyze File"):
    file = sl.file_uploader("Upload file a PDF file", type ="pdf")
    if file: #if file has been uploaded
        reader = PdfReader(file)
        num_pages = len(reader.pages)

        sentences = []
        for page in reader.pages:
            text = page.extract_text()
            sentences.extend(text.split(". "))
        
        df = pd.DataFrame(sentences, columns = ["Sentence"])
        df["Score"] = df["Sentence"].apply(score)
        sl.write(f"Number of Pages: {num_pages}")
        sl.write(f"Number of Sentences: {len(df)}")

        sl.write("Sentence Scores:")
        sl.write(df)
