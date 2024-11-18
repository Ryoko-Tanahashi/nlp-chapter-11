import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from janome.tokenizer import Tokenizer
t = Tokenizer(wakati=True)

# 11.3 データセットの前処理
# ボキャブラリの作成
def build_vocabulary(texts, num_words=None):
    # num_words: 出現頻度の高い上位num_words個の単語のみを保持（今回は指定していないため、すべての単語が語彙に含まれる）
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token='<UNK>', filters='')
    # テキストデータのトークン化を行う。Out-Of-Vocabulary（未知語）の場合は、<UNK>に置き換える。句読点などの記号は除去しない。
    tokenizer.fit_on_texts(texts)
    # 単語ごとにインデックスを割り当てる
    return tokenizer

def tokenize(text):
    # 日本語の分かち書きを行う
    return t.tokenize(text)

# 開始記号と終了記号の付与
def preprocess_dataset(texts):
    return ['<start> {} <end>'.format(text) for text in texts]

# 日本語の前処理
def preprocess_ja(texts):
    # 分かち書きがされたテキストを空白文字で連結する
    # 「私はりんごが好きです」→「私 は りんご が 好き です」
    return [' '.join(tokenize(text)) for text in texts]

# 入力用データセットの作成
def create_dataset(en_texts, ja_texts, en_vocab, ja_vocab):
    # テキストデータを単語IDの系列に変換する
    en_seqs = en_vocab.texts_to_sequences(en_texts)
    ja_seqs = ja_vocab.texts_to_sequences(ja_texts)
    # 系列の長さを揃える
    en_seqs = pad_sequences(en_seqs, padding='post')
    ja_seqs = pad_sequences(ja_seqs, padding='post')

    return [en_seqs, ja_seqs[:, :-1]], ja_seqs[:, 1:]


