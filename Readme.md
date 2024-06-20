# Project Overview
The PDF Question Answering System is a web application developed to extract information from PDF documents and provide accurate answers to user queries. Leveraging natural language processing (NLP) techniques and the Flask web framework, the system allows users to upload PDFs, input queries, and receive relevant answers based on the content of the documents.
# Installations
To Run the Pogramme in your system you need to: 
- clone the repo 
```bash
git clone https://github.com/PulkitRawat/Pdf-AnsweringAi.git
cd  Pdf-AnsweringAi
```
- install the required packages
```bash
pip intall -r requirments.txt
```
# Usage
To use the PDF Question Answering System:
- Run the application
```bash
python deployment.py
```
- Upload a PDF document (scanned or unscanned).
- Specify the type of PDF.
- Enter a query related to the document content.
- Submit the query and receive the answer on the results page.
# Project Structure
The project is structured as follows:
* bert_ model.py: Contains the logic for text extraction, preprocessing, sentence embedding using BERT, similarity calculation, and answer generation.
- deployment.py: Implements the Flask web framework for creating routes, handling HTTP requests, and rendering HTML templates.
- templates/: Directory containing HTML files for the web interface, including pages for uploading PDFs, entering queries, and displaying answers

# Dependencies
- Flask
- torch
- transformers
- numpy
- scikit-learn
- nltk
- PyMuPDF
- pytesseract
- torchvision
