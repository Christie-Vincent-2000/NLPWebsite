from flask import Flask, request, render_template,jsonify
import nltk
import nltk
from string import punctuation
from nltk.corpus import stopwords
from rake_nltk import Rake
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from autocorrect import spell
import os

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/keyExt', methods=["GET"])
def keyword_extraction():
    return render_template('keyExt.html')


@app.route('/preproc', methods=["GET"])
def pre_process():
    return render_template('preproc.html')


@app.route('/others', methods=["GET"])
def others():
    return render_template('others.html')


@app.route('/summary', methods=["GET"])
def summary():
    return render_template('text_Summarization.html')


@app.route('/installation', methods=["GET"])
def installation():
    return render_template('installation.html')


@app.route('/lower', methods=["GET", "POST"])
def lower_case():
    text1 = request.form['text']
    word = text1.lower()
    result = {
        "result": word
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/upper', methods=["GET", "POST"])
def upper_case():
    text1 = request.form['text']
    word = text1.upper()
    result = {
        "result": word
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/sent_tokenize', methods=["GET", "POST"])
def sent_tokenize():
    text = request.form['text']
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.sentences_from_text(text)
    sentences = [sentence.split() for sentence in sentences]
    sentences = [[word.strip(",.?!") for word in sentence]
                    for sentence in sentences]
    result = {
        "result ": str(sentences) #remove str() if you want the output as list
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/word_tokenize', methods=["GET", "POST"])
def word_tokenize():
    text = request.form['text']
    word_tokenize = wordpunct_tokenize(text)
    result = {
        "result": str(word_tokenize) #remove str() if you want the output as list
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/spell_check', methods=["GET", "POST"])
def spell_check():
    text = request.form['text']
    spells = [spell(w) for w in (nltk.word_tokenize(text))]
    result = {
        "result": " ".join(spells)
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/lemmatize', methods=["GET", "POST"])
def lemmatize():
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()

    text = request.form['text']
    word_tokens = nltk.word_tokenize(text)
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in
                       word_tokens]
    result = {
        "result": " ".join(lemmatized_word)
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/stemming', methods=["GET", "POST"])
def stemming():
    from nltk.stem import SnowballStemmer
    snowball_stemmer = SnowballStemmer('english')

    text = request.form['text']
    word_tokens = nltk.word_tokenize(text)
    stemmed_word = [snowball_stemmer.stem(word) for word in word_tokens]
    result = {
        "result": " ".join(stemmed_word)
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/remove_tags', methods=["GET", "POST"])
def remove_tags():
    import re
    text = request.form['text']
    cleaned_text = re.sub('<[^<]+?>', '', text)
    result = {
        "result": cleaned_text
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


@app.route('/remove_numbers', methods=["GET", "POST"])
def remove_numbers():
    text = request.form['text']
    remove_num = ''.join(c for c in text if not c.isdigit())
    result = {
        "result": remove_num
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/remove_punct', methods=["GET", "POST"])
def remove_punct():
    from string import punctuation
    def strip_punctuation(s):
        return ''.join(c for c in s if c not in punctuation)

    text = request.form['text']
    text = strip_punctuation(text)
    result = {
        "result": text
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/remove_stopwords', methods=["GET", "POST"])
def remove_stopwords():
    from nltk.corpus import stopwords
    stopword = stopwords.words('english')
    text = request.form['text']
    word_tokens = nltk.word_tokenize(text)
    removing_stopwords = [word for word in word_tokens if word not in stopword]
    result = {
        "result": " ".join(removing_stopwords)
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


@app.route("/keyword", methods=["GET","POST"])
def keyword():
    text = request.form['text']
    rake_nltk_var = Rake()
    rake_nltk_var.extract_keywords_from_text(text)
    keyword_extracted = rake_nltk_var.get_ranked_phrases()
    result = {
        "result": keyword_extracted
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


@app.route("/summarize", methods=["GET","POST"])
def summarize():
    text = request.form['text']
    import string
    from heapq import nlargest
    # If the length of the text is greater than 20, take a 10th of the sentences
    if text.count(". ") > 20:
     length = int(round(text.count(". ")/10, 0))
    # Otherwise return five sentences
    else:
        length = 1
    # Remove punctuation
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    # Remove stopwords
    processed_text =[word for word in nopunc.split() if word.lower() not in nltk.corpus.stopwords.words('english')]
    # Create a dictionary to store word frequency
    word_freq = {}
    # Enter each word and its number of occurrences
    for word in processed_text:
        if word not in word_freq:
            word_freq[word] = 1
        else:
            word_freq[word] = word_freq[word] + 1
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = (word_freq[word]/max_freq)
    # Create a list of the sentences in the text
    sent_list = nltk.sent_tokenize(text)
    # Create an empty dictionary to store sentence scores
    sent_score = {}
    for sent in sent_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_freq.keys():
                if sent not in sent_score.keys():
                    sent_score[sent] = word_freq[word]
                else:
                    sent_score[sent] = sent_score[sent] + word_freq[word]
    summary_sents = nlargest(length, sent_score, key = sent_score.get)
    summary = ' '.join(summary_sents)
    result = {
        "result": summary
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

