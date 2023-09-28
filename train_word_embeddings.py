import os
from gensim.models import FastText
import string

punctuations = set(string.punctuation)


def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def preprocess(myfile):
    with open(myfile, encoding='utf-8') as f:
        text_lines  = f.readlines()
    n_tokens = 0
    new_text_lines = []
    token_set = []
    for text in text_lines:
        text_tokens = text.split()
        new_text_tokens = [word for word in text_tokens if word not in punctuations]
        new_text_lines.append(new_text_tokens)
        n_tokens+=len(new_text_tokens)
        token_set+=new_text_tokens

    print("# sentences", len(new_text_lines))
    print("# Tokens ", n_tokens)
    print("# Vocabulary ", len(set(token_set)))

    return new_text_lines


#https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#
#https://radimrehurek.com/gensim/models/word2vec.html#module-gensim.models.word2vec
#https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html
def train_fastText(lang):
    model_full = FastText(preprocess('data/'+lang+'.all'), vector_size=300, window=5, min_count=3, workers=4, sg=1, epochs=10, negative=10)
    output_dir = "embeddings/"+lang+"/"
    create_dir(output_dir)
    model_full.wv.save(output_dir+lang+".bin")
    print("embedding training Done")

if __name__ == '__main__':
    #languages = ['luo', 'ewe', 'ibo', 'kin', 'lug', 'mos', 'pcm', 'zul']
    languages = ['yo']
    #languages = ['sna', 'tsn', 'swa']
    for lang in languages:
        train_fastText(lang)
