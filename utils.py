from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu

# 11.3 データセットの準備
def load_dataset(filename):
    en_texts = []
    ja_texts = []
    with open(filename, encoding='utf-8') as f:
        for line in f: #各行について
            en_text, ja_text = line.strip().split('\t')[:2] #タブで分割し、最初の2つを取り出す
            en_texts.append(en_text)
            ja_texts.append(ja_text)
    return en_texts, ja_texts

# 11.3 評価関数の実装
def evaluate_bleu(X, y, api):
    d = defaultdict(list)
    for source, target in zip(X, y):
        d[source].append(target)
    hypothesis = []
    references = []

    for source, targets in d.items():
        pred = api.predict(source)
        # 翻訳仮説と正解文をリストに追加
        hypothesis.append(pred)
        references.append(targets)

    # BLEUスコアの計算
    bleu_score = corpus_bleu(references, hypothesis)
    return bleu_score