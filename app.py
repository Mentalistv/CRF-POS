import re
import streamlit as st
import pickle

def word_features(sentence, i, vocab):
	word = sentence[i][0]
	features = {
		'word': word,
		'is_first': i == 0, #if the word is a first word
		'is_last': i == len(sentence) - 1, #if the word is a last word
		'is_capitalized': word[0].upper() == word[0],
		'is_all_caps': word.upper() == word,	 #word is in uppercase
		'is_all_lower': word.lower() == word,	 #word is in lowercase
		#prefix of the word
		'prefix-1': word[0],
		'prefix-2': word[:2],
		'prefix-3': word[:3],
		#suffix of the word
		'suffix-1': word[-1],
		'suffix-2': word[-2:],
		'suffix-3': word[-3:],
		#extracting previous word
		'prev_word': '' if i == 0 else sentence[i-1][0],
		#extracting next word
		'next_word': '' if i == len(sentence)-1 else sentence[i+1][0],
		'has_hyphen': '-' in word, #if word has hypen
		'is_numeric': word.isdigit(), #if word is in numeric
		'capitals_inside': word[1:].lower() != word[1:],
        'is_unknown': word not in vocab  # Flag if the word is not in the known vocabulary - CHANGE
	}
	return features


def tokenize(sentence):
    tokens = re.findall(r"[\w]+|[.,!?;]", sentence)
    return tokens

def sent2features(sentence, vocab):
    return [word_features(sentence, i, vocab) for i in range(len(sentence))]


def predict(sent, model):
    sent  = sent.split()
    pred = model._viterbi(sent)
    return pred






vocab = []
with open("vocab.txt", 'r') as f:
    vocab = [line.rstrip('\n') for line in f]

crf_model = pickle.load(open("crf_model.pkl", 'rb')) 


predict("Hi my name  is Harsh", crf_model)


tag_list = ['DET', 'PRT', 'ADV', 'X', 'CONJ', 'ADJ', 'ADP', 'PRON', 'NOUN', '.', 'NUM', 'VERB']

# Streamlit interface
st.title("POS Tagging with CRF")

# Input text
sentence = st.text_area("Enter a sentence:")

if st.button("Predict Tags"):
    if sentence:
        t,p = predict(sentence, crf_model)
        st.write("Word : Tag")
        for i in range(len(t)):
            st.write(t[i],':',p[i])
