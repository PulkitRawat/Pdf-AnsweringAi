from flask import Flask, request, render_template
import os
from bert_model import * 
from nltk.tokenize import sent_tokenize

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        question = request.form['question']
        file_type = request.form['file_type']
        pdf_text = extract_text(file_type[0], file_path)
        sentences = sent_tokenize(pdf_text)
        sentence_embeddings = [get_sentence_embedding(sentence) for sentence in sentences]
        # relevant_sections = rrs(question, sentences, sentence_embeddings, model)
        answer = find_answer(question, sentences, sentence_embeddings)
        return render_template('result.html', question=question, answer=answer)
    
if __name__ == '__main__':
    app.run(debug=True)

    