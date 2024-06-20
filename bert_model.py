import numpy as np
import torch
import fitz
import pytesseract
import string
import io

from transformers import BertTokenizer, BertModel, BertForQuestionAnswering
from PIL import Image 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Text Extraction
def extract_text(t_pdf:str, path:str) -> str:
    '''
    Extract text from the PDF.

    Args:
        t_pdf (str): Type of PDF ('s' for scanned, 'u' for unscanned).
        path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    '''

    document = fitz.open(path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        if t_pdf == "s":
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            text += pytesseract.image_to_string(img)
        else:
            text += page.get_text()
    return text
    
# Preprocess of Text
def preprocess(text: str):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    sentences = get_clean_text(text)
    s_embeddings = [get_sentence_embedding(sentence, tokenizer, model) for sentence in sentences]
    return s_embeddings
## Text Cleaning 
def get_clean_text(text:str)->list[str]:
    '''
    Clean the extracted text by removing stopwords and punctuation.
    
    Args:
        text(str): input pdf_text to be cleaned.
        
    Returns:
        list(str): list of sentences of cleaned text.
    '''
    sentences = sent_tokenize(text)
    c_sentences = []
    stop_words = set(stopwords.words("english"))
    punctuation = set(string.punctuation)
    for sentence in sentences:
        sentence = sentence.lower()
        words = word_tokenize(sentence)
        words = [word for word in words if word not in stop_words and word not in punctuation]
        c_sentence = " ".join(words)
        c_sentences.append(c_sentence)
    return c_sentences
## Text Embedding
def get_sentence_embedding(sentence:str, tokenizer=tokenizer, model = model)->torch.Tensor:
    '''
    Generate the sentence embedding in form of tensor.
    
    Args:
        sentence(str): input sentence to generate the embedding for.
        tokenizer(optional): the model used to tokenize the input sentence.
            Default to BertTokenizer.from_pretrained("bert-base-uncased").
        model(optional): the embedding model use to generate embedding.
            Default to BertModel.from_pretrained("bert-base-uncased").        
    Returns:
        torch.Tensor: sentence embedding.
    '''
    inputs = tokenizer(sentence, return_tensors = "pt", truncation = True, max_length = 512)
    with torch.no_grad():
        output = model(**inputs)
    sentence_embedding = output.last_hidden_state.mean(dim=1).squeeze()
    return sentence_embedding

# Retrieve Relevant section
def rrs(question: str, sentences: list[str], sentence_embeddings: torch.Tensor, top_n)-> list[str]:
    '''
    Retrieving the sentences with the closest relationship i.e. relevant to the asked question.
    
    Args:
        question(str): question asked.
        sentences(list[str]): sentences of pdf text.
        sentences_embeddings(torch.Tensor): embedding of the sentences of the pdf text.
        top_n(int): no. of sentences to be taken during selection of relevant section.

    Returns:
        list[str]:list of relevant sentences.
    '''
    # question_embedding = preprocess(question)[0].unsqueeze(0)
    question_embedding = get_sentence_embedding(question, tokenizer, model).unsqueeze(0)
    similarities = cosine_similarity(question_embedding, torch.stack(sentence_embeddings).numpy()) 
    m_relevant_indices = np.argsort(similarities[0])[::-1][:top_n]
    relevant_sentences = [sentences[i] for i in m_relevant_indices]
    return relevant_sentences

# Finding the Answer
def find_answer(question, sentences, sentence_embedding, top_n = 10)-> str:
    '''
    Generating the final answer from the relevant section.
    
    Args:
        question(str): question asked.
        sentences(list[str]): sentences of pdf text.
        sentences_embeddings(torch.Tensor): embedding of the sentences of the pdf text.
        top_n(int): no. of sentences to be taken during selection of relevant section.

    Returns:
        str: answer to the question.
    '''
    relevant_sections = rrs(question, sentences, sentence_embedding, top_n)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    relevant_section = " ".join(relevant_sections)
    inputs = tokenizer(question, relevant_section, return_tensors='pt', truncation =True, max_length = 512)
    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits) + 1
    answer_span = inputs['input_ids'][0][start_index:end_index]
    answer = tokenizer.decode(answer_span)
    return answer

