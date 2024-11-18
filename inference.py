import numpy as np

class InferenceAPI:
    def __init__(self, encoder_model, decoder_model, en_vocab, ja_vocab):
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.en_vocab = en_vocab
        self.ja_vocab = ja_vocab

    def predict(self, text):
        # 入力文を固定長のベクトルに変換
        output, state = self._compute_encoder_output(text)
        # デコーダを用いて日本語の単語に対応するIDを生成
        sequence = self._generate_sequence(output, state)
        # IDを日本語の単語に変換
        decoded = self._decode(sequence)
        return decoded
    
    def _compute_encoder_output(self, text):
        x = self.en_vocab.texts_to_sequences([text])
        output, state = self.encoder_model.predict(x)
        return output, state
    
    def _compute_decoder_output(self, target_seq, state, enc_output=None):
        output, state = self.decoder_model.predict([target_seq, state])
        return output, state
    
    def _generate_sequence(self, enc_output, state, max_seq_len=50):
        target_seq = np.array([[self.ja_vocab.word_index['<start>']]])
        sequence = []
        for _ in range(max_seq_len):
            output, state = self._compute_decoder_output(target_seq, state, enc_output)
            sampled_token_index = np.argmax(output[0, -1, :])
            if sampled_token_index == self.ja_vocab.word_index['<end>']:
                break
            sequence.append(sampled_token_index)
            target_seq = np.array([[sampled_token_index]])
        return sequence
    
    def _decode(self, sequence):
        decoded = self.ja_vocab.sequences_to_texts([sequence])
        decoded = decoded[0].split(' ')
        return decoded
    
# 11.5 アテンション付き予測用クラスの定義
class InferenceAPIforAttention(InferenceAPI):
    def _compute_decoder_output(self, target_seq, state, enc_output=None):
        output, state = self.decoder_model.predict([target_seq, enc_output, state])
        return output, state
    
