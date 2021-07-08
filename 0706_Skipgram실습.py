from google.colab import drive
drive.mount('/content/drive')

# skipgram 실습 과제
# 점심 이후

# (1) 영어소설 corpus, Gutenburg corpus => 영어소설 10개
# (2) working, worked -> 1단어인 work로 사용 => stemmer 사용
# (3) vocavulary 생성
# (4) 소설 문장 1줄씩 읽어서 tri-gram 생성(nltk 실습 참조) 
# -> i love you very much -> (i love you),(love you very)...
# (5) tri-gram 으로 학습 데이터 생성

# (6) 각 단어를 vocabulary 의 index로 표현(ex.love = vocab 32번째) 한 후 
# one-hot encoding. love = [0000...1(32번째)00...]
# (7) 네트워크 구성(vocab의 사이즈만큼). hidden layer = (word vector size = 32개 크기로 표현)
# (8) 단어의 one-hot 값을 입력 -> 은닉층 출력(32)
# (9) father = [dddd...._32개], mother = [], doctor = [] -> sine similarity 계산 
# (10) father + mother = []

# 입력층_embadding layer  출력층_sparse categorical(데이터 많기 때문에 to categorical 쓰지 말기)

import numpy as np
import pandas as pd
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('punkt')
nltk.download('gutenberg')

text_id = nltk.corpus.gutenberg.fileids()
text_id

n = 10
for i, text_id in enumerate(nltk.corpus.gutenberg.fileids()[:n]):
    text = nltk.corpus.gutenberg.raw(text_id)
    sentences = nltk.sent_tokenize(text)

print("문장 개수 = ", len(text))

sentences = list(nltk.sent_tokenize(text))

sentences[:5]

nltk.download('wordnet')
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

for word in sentences:
  print(stemmer.stem(word))

# 사전 (vocabulary) 생성
word2idx = {}
n_idx = 0
for word in sentences:
    if words.lower() not in word2idx:
        word2idx[word.lower()] = n_idx
        n_idx += 1
# idx2word = {v:k for k, v in word2idx.items()}

word2idx.items() #.items() -> 딕셔너리의 item으로 출력해줌

word2idx = {}
n_idx = 0
for word in sentences:
    if word.lower() not in word2idx:
      word2idx[word.lower()] = n_idx
      n_idx  += 1
idx2word = {v:k for k,v in word2idx.items()}

word2idx.items()

# trigram
trigram = [(a, b, c) for a, b, c in nltk.trigrams(word2idx)] # 세 개의 연속된 단어쌍
trigram[:5]

# 학습 데이터 생성

# 문장을 단어의 인덱스로 표현
tokenizer = Tokenizer()
sent_idx = tokenizer.texts_to_sequences(word2idx)
sent_idx[0]

def get_context(x, count = True):
    idx = word2idx[x]
    word_count = {v:0 for k,v in word2idx.items()}
    for s_idx in sent_idx:
        if idx in s_idx:
            for i in s_idx:
                word_count[i] += 1

    result = sorted(word_count.items(), key=(lambda x: x[1]), reverse=True)
    result = [(idx2word[i], c) for i, c in result[:20]]

    if count:
        return result
    else:
        return [w for w, c in result]

#

# # 사전 (vocabulary) 생성 
# # word2inx - 단어를 숫자로 변환
# # idx2word - 숫자 단어로 변환
# # nlp 필수사항

# word2idx = {} # 키 순서는 랜덤. 알파벳 순서로 보여줌.
# n_idx = 0
# for sent in sentences: # all_tokens 불용어 처리 된 텍스트
#     for word in sent:
#         if word.lower() not in word2idx: # word to index
#             word2idx[word.lower()] = n_idx # word2inx['love'] = 32, 32 들어감. 
#             n_idx += 1
# idx2word = {v:k for k, v in word2idx.items()} 

# # 키와 벨류 받아서 v:k 로 바꿔줌
# #idx2word[0] = 'natural' 이런 형식으로 

# word2idx
# word2idx.items() # .items() -> 딕셔너리의 item으로 출력해줌

# idx2word[15]

# word2idx = {}
# n_idx = 0
# for sent in all_tokens :
#   for word in sent :
#     if word.lower() not in word2idx:
#       word2idx[word.lower()] = n_idx
#       n_idx  += 1
# idx2word = {v:k for k,v in word2idx.items()}

# word2idx.items()

# word_tok = nltk.word_tokenize(sentence)
# print(word_tok)

# nltk.pos_tag(word_tok)

# # 명사와 형용사만 표시한다.
# sent_nnjj = [word for word, pos in nltk.pos_tag(word_tok) if pos == 'NN' or pos == 'JJ']  # 단어, 품사 나타나게 함

# sent_nnjj

# trigram
trigram = [(a, b, c) for a, b, c in nltk.trigrams(sentences)] # 세 개의 연속된 단어쌍
trigram

# 텍스트에서 단어 분리
word_tokens = [nltk.word_tokenize(x) for x in nltk.sent_tokenize(sentences)] # 많이 쓰임
word_tokens[0] # 2차원 구조 리스트

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# 특정 파일의 텍스트 문서를 조회한다.
text = nltk.corpus.gutenberg.raw('austen-emma.txt')   # 원시 형태 raw 데이터
print(text[:600])
print("문자 개수 = ", len(text))

import numpy as np
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('punkt')
nltk.download('stopwords')  # 불용어 목록
nltk.download('gutenberg')
text_id = nltk.corpus.gutenberg.fileids()
text_id

  for i in gutenberg 

# 특정 파일의 텍스트 문서를 조회한다.
text = nltk.corpus.gutenberg.raw('austen-emma.txt')
print(text[:600])
print("문자 개수 = ", len(text))

sentences = nltk.sent_tokenize(text) # 문장들이 리스트 형태로 있음 
sentences[:5] # 앞 다섯개 문장

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) # 함수가 아닌 member function

# 특정 문서를 문장 단위로 읽어온다.

n = 5

sentence = nltk.corpus.gutenberg.sents('austen-emma.txt')
for i in n range(5):
    print(sentence[i])
print("문장 개수 = ", len(sentence))

import nltk
from nltk.corpus import webtext
nltk.download('punkt')
nltk.download('webtext')

# Webtext 코퍼스의 파일 ID를 조회한다.
textId = webtext.fileids()
print(textId)

text = """
Natural language processing (NLP) is a subfield of computer science, information engineering, 
and artificial intelligence concerned with the interactions between computers and human (natural) languages, 
in particular how to program computers to process and analyze large amounts of natural language data. 
Challenges in natural language processing frequently involve speech recognition, natural language understanding, 
and natural language generation.
"""

sent_tok = nltk.sent_tokenize(text) # 문서 -> 문장. 위 문서를 두 문장('.'으로 나뉨)으로 나눔
print(len(sent_tok))

sent_tok[1] # 첫 번째 문장
len(sent_tok) # 문장의 개수

# 텍스트에서 단어 분리
word_tokens = [nltk.word_tokenize(x) for x in nltk.sent_tokenize(text)] # 많이 쓰임
word_tokens[0] # 2차원 구조 리스트

# stop words 제거 
# 덜 중요한 단어들('.') = 불용어(stopwords)

stopwords = nltk.corpus.stopwords.words('english')  # nltk 등록된 stop word
stopwords.append(',') # 리스트 형태 .append - 등록
stopwords

all_tokens = []
for sent in word_tokens: # 불용어를 제거. 텍스트를 문장으로 쪼갬 ,그 후 다시 단어로 쪼갬
    all_tokens.append([word for word in sent if word.lower() not in stopwords]) 
all_tokens

# Stemming
# 동사의 원형으로 변환

from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

for word in ['working', 'works', 'worked']: 
# 뒤에 어미(surfix) 변환. 그러나 의미 정보가 유실. 이후에는 의미 분석 -> subword tokenize
    print(stemmer.stem(word))

stemmer = LancasterStemmer()

for word in ['working','works','worked']:
  print(stemmer.stem(word))

# 사전 (vocabulary) 생성 
# word2inx - 단어를 숫자로 변환
# idx2word - 숫자 단어로 변환

# nlp 필수사항

word2idx = {} # 키 순서는 랜덤. 알파벳 순서로 보여줌.
n_idx = 0
for sent in all_tokens:  # all_tokens 불용어 처리 된 텍스트
    for word in sent:
        if word.lower() not in word2idx: # word to index
            word2idx[word.lower()] = n_idx # word2inx['love'] = 32, 32 들어감. 
            n_idx += 1
idx2word = {v:k for k, v in word2idx.items()} 

# 키와 벨류 받아서 v:k 로 바꿔줌
#idx2word[0] = 'natural' 이런 형식으로 

word2idx
word2idx.items() # .items() -> 딕셔너리의 item으로 출력해줌
idx2word[15]

word2idx = {}
n_idx = 0
for sent in all_tokens :
  for word in sent :
    if word.lower() not in word2idx:
      word2idx[word.lower()] = n_idx
      n_idx  += 1
idx2word = {v:k for k,v in word2idx.items()}

word2idx.items()

# trigram
trigram = [(a, b, c) for a, b, c in nltk.trigrams(sent_nnjj)] # 세 개의 연속된 단어쌍
trigram

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Colab Notebooks
import konlpy
from konlpy.tag import Okt
import nltk

nltk.download('punkt')
nltk.download('stopwords')  # 불용어 목록

f = open("data/ko_novel.txt", 'r')
text = f.read()

def get_context(search_word, text ,top_n=30):
  cnt={}
  sentences = nltk.sent_tokenize(text)
  okt = Okt()
  for sentence in sentences:
    sent_nnjj = [word for word, pos in okt.pos(sentence) if pos == 'Adjective' or pos == 'Noun']
    try:
      bigram = [(a, b) for a, b in nltk.bigrams(sent_nnjj)]
      for i in bigram:
        if i[0] == search_word:
          if cnt.get(i[1]) ==None:
            cnt[i[1]] = 1
          else:
            cnt[i[1]] +=1
        elif i[1] == search_word:
          if cnt.get(i[0]) ==None:
            cnt[i[0]] = 1
          else:
            cnt[i[0]] +=1
        else:
          continue
    except:
      continue
  sorted_cnt = sorted(cnt, key=cnt.get, reverse=True)[:top_n]
  for i in sorted_cnt:
    print(i, cnt[i])
  return cnt

father = get_context('아버지',text,30)

mother = get_context('어머니',text,30)

dortor = get_context('의사',text,30)

def jaccard(x, y):
    hab = set(x) | set(y)
    gyo = set(x) & set(y)
    return len(gyo) / len(hab)

father_keys = father.keys()
mother_keys = mother.keys()
doctor_keys = dortor.keys()

jaccard(father_keys, mother_keys)#엄마아빠
jaccard(father_keys, doctor_keys)#아빠의사
