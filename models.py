from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Embedding, GRU, Dense

# 11.3 ベースモデルの定義
class BaseModel :
    # モデルの構築
    def build(self):
        # サブクラスで実装するため、エラーを発生させる
        raise NotImplementedError()
    
    # モデルアーキテクチャの保存
    def save_as_json(self, filepath):
        model = self.build()
        with open(filepath, 'w') as f:
            f.write(model.to_json())

    @classmethod
    # モデルの読み込み
    def load(cls, architecture_file, weights_file, by_name=True):
        with open(architecture_file) as f:
            model = model_from_json(f.read())
            model.load_weights(weights_file, by_name=by_name)
        return model

# 11.3 エンコーダの定義
class Encoder(BaseModel):
    # インスタンスの初期化
    def __init__(self, input_dim, emb_dim=300, hid_dim=256, return_sequences=False):
        # 任意の長さの入力を受け付ける
        self.input = Input(shape=(None,), name='encoder_input')
        # 単語を連続ベクトルに変換する埋め込み層。0はパディング用のため、マスクする
        self.embedding = Embedding(input_dim=input_dim, output_dim=emb_dim, mask_zero=True, name='encoder_embedding')
        # GRUレイヤーの定義。return_sequencesがTrueの場合、全時刻の出力を返す。return_stateがTrueの場合、最終時刻の隠れ状態も返す
        self.gru = GRU(hid_dim, return_sequences=return_sequences, return_state=True, name='encoder_gru')

    def __call__(self):
        # 入力データを受け取る
        x = self.input
        # 入力データを埋め込みベクトルに変換
        embedding = self.embedding(x)
        # GRUレイヤーに入力データを与え、出力と最終時刻の隠れ状態を取得
        output, state = self.gru(embedding)
        return output, state

    def build(self):
        # __call__メソッドを呼び出し、出力と最終時刻の隠れ状態を取得
        output, state = self()
        # モデルを構築
        return Model(inputs=self.input, outputs=[output, state])
    
# 11.3 デコーダの定義
class Decoder(BaseModel):
    def __init__(self, output_dim, emb_dim=300, hid_dim=256):
        self.input = Input(shape=(None,), name='decoder_input')
        self.embedding = Embedding(input_dim=output_dim, output_dim=emb_dim, mask_zero=True, name='decoder_embedding')
        self.gru = GRU(hid_dim, return_sequences=True, return_state=True, name='decoder_gru')
        # 出力層の活性化関数をsoftmax関数に設定し、各単語の出現確率を計算
        self.dense = Dense(output_dim, activation='softmax', name='decoder_output')

        # エンコーダの隠れ状態を受け取る
        self.state_input = Input(shape=(hid_dim,), name='decoder_state_in')

    def __call__(self, states, enc_output=None):
        x = self.input
        embedding = self.embedding(x)
        outputs, state = self.gru(embedding, initial_state=states)
        outputs = self.dense(outputs)
        return outputs, state

    def build(self):
        decoder_output, decoder_state = self(states=self.state_input)
        return Model(inputs=[self.input, self.state_input], outputs=[decoder_output, decoder_state])
    
# 11.3 モデル全体の定義
class Seq2Seq(BaseModel):
    # エンコーダとデコーダの統合
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def build(self):
        # エンコーダを実行し、出力と最終時刻の隠れ状態を取得
        encoder_output, state = self.encoder()
        # stateを初期状態としてデコーダを実行し、出力を取得
        decoder_output, _ = self.decoder(states=state, enc_output=encoder_output)
        # エンコーダとデコーダの入力を統合したモデルを作成し、予測シーケンスを生成
        return Model([self.encoder.input, self.decoder.input], decoder_output)
    
# 11.5 アテンションの計算の実装
from tensorflow.keras.layers import Dot, Activation, Concatenate

class LuongAttention:
    def __init__(self, units=300):
        self.dot = Dot(axes=[2, 2], name='dot')
        self.attention = Activation('softmax', name='attention')
        self.context = Dot(axes=[2, 1], name='context')
        self.concat = Concatenate(name='concat')
        self.fc = Dense(units, activation='tanh', name='attn_out')

    def __call__(self, enc_output, dec_output):
        attention = self.dot([dec_output, enc_output])
        attention_weight = self.attention(attention) #重みベクトル
        context_vector = self.context([attention_weight, enc_output]) #文脈ベクトル
        concat_vector = self.concat([context_vector, dec_output])
        output = self.fc(concat_vector) #予測用ベクトル
        return output
    

# 11.5 アテンション付きデコーダの定義
class AttentionDecoder(Decoder):
    def __init__(self, output_dim, emb_dim=300, hid_dim=256):
        super().__init__(output_dim, emb_dim, hid_dim)
        self.attention = LuongAttention()
        self.enc_output = Input(shape=(None, hid_dim), name='encoder_output')

    def __call__(self, states, enc_output=None):
        x = self.input
        embedding = self.embedding(x)
        outputs, state = self.gru(embedding, initial_state=states)
        outputs = self.attention(enc_output, outputs) #アテンションの計算
        outputs = self.dense(outputs)
        return outputs, state
    
    def build(self):
        decoder_output, decoder_state = self(states=self.state_input, enc_output=self.enc_output)
        return Model(inputs=[self.input, self.enc_output, self.state_input], outputs=[decoder_output, decoder_state]) #エンコーダの出力を追加