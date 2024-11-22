import pickle
import re
import spacy



if __name__ == '__main__':

    nlp = spacy.load('en_core_web_sm')
    sentence = " I'm   running   ![LAugHter]  "
    sentence = re.sub('\s+',' ', sentence.strip())
    print(sentence)

    for token in nlp(sentence):
        print(token.text.lower())
