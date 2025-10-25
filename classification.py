import numpy as np
import tensorflow.lite as tflite
import librosa
import json
import pickle
from vosk import Model, KaldiRecognizer
from transformers import AutoTokenizer, AutoModel
import torch
import os
from dotenv import load_dotenv
import requests

load_dotenv()

class AudioTextClassifier:
    def __init__(self, input_path, use_s3 = False, s3_url = None):
        if use_s3:
            if not s3_url:
                raise ValueError("s3_url must be provided when use_s3=True")
            self._download_models_from_s3(s3_url, input_path)

        self.interpreter_audio = tflite.Interpreter(model_path=f"{input_path}/audio_prediction_model.tflite")
        self.interpreter_audio.allocate_tensors()
        self.interpreter_text = tflite.Interpreter(model_path=f"{input_path}/text_prediction_model.tflite")
        self.interpreter_text.allocate_tensors()

        model_name = f"{input_path}/encoding_model"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoding_model = AutoModel.from_pretrained(model_name)

        with open(f"{input_path}/label_encoder.pkl", "rb") as f:
            self.le = pickle.load(f)
        
        self.le_classes = self.le.classes_
        self.N_MFCC = 40
        self.MAX_LEN = 174
        self.SR = 22050

        self.vosk_model = Model(f"{input_path}/vosk_model/vosk-model-small-en-us-0.15")

        self.emotions = ["neutral_calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        self.emo2idx = {emo: i for i, emo in enumerate(self.emotions)}
        self.w = np.array([0.0, 33.0, 66.0, 100.0], dtype=np.float32)

        self.M = np.load(f"{input_path}/M_fusion.npy")
        self.b = np.load(f"{input_path}/b_fusion.npy")
    
    def _download_models_from_s3(self, s3_base_url, local_path):
        os.makedirs(local_path, exist_ok=True)
        
        files_to_download = [
            "audio_prediction_model.tflite",
            "text_prediction_model.tflite",
            "label_encoder.pkl",
            "M_fusion.npy",
            "b_fusion.npy"
        ]
        
        for filename in files_to_download:
            local_file = os.path.join(local_path, filename)
            if not os.path.exists(local_file):
                url = f"{s3_base_url}/{filename}"
                print(f"Downloading {filename}...")
                self._download_file(url, local_file)
        
        encoding_model_path = os.path.join(local_path, "encoding_model")
        os.makedirs(encoding_model_path, exist_ok=True)
        
        encoding_files = [
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt"
        ]
        
        for filename in encoding_files:
            local_file = os.path.join(encoding_model_path, filename)
            if not os.path.exists(local_file):
                url = f"{s3_base_url}/encoding_model/{filename}"
                try:
                    self._download_file(url, local_file)
                except Exception as e:
                    print(f"Note: {filename} not found (might be optional): {e}")
        
        vosk_model_path = os.path.join(local_path, "vosk_model", "vosk-model-small-en-us-0.15")
        os.makedirs(vosk_model_path, exist_ok=True)
        
        vosk_files = [
            "am/final.mdl",
            "conf/mfcc.conf",
            "conf/model.conf",
            "graph/disambig_tid.int",
            "graph/Gr.fst",
            "graph/HCLG.fst",
            "graph/phones/word_boundary.int",
            "ivector/final.dubm",
            "ivector/final.ie",
            "ivector/final.mat",
            "ivector/global_cmvn.stats",
            "ivector/online_cmvn.conf",
            "ivector/splice.conf"
        ]
        
        for filename in vosk_files:
            local_file = os.path.join(vosk_model_path, filename)
            if not os.path.exists(local_file):
                url = f"{s3_base_url}/vosk_model/vosk-model-small-en-us-0.15/{filename}"
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                try:
                    self._download_file(url, local_file)
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
    
    def _download_file(self, url, local_path):
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"  ✓ {os.path.basename(local_path)}")

    def _extract_features(self, file_path):
        y, sr = librosa.load(file_path, sr=self.SR)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.N_MFCC)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        combined = np.vstack([mfcc, delta, delta2]) 

        if combined.shape[1] < self.MAX_LEN:
            pad_width = self.MAX_LEN - combined.shape[1]
            combined = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')
        else:
            combined = combined[:, :self.MAX_LEN]

        return combined

    def _predict_audio(self, file_path):
        features = self._extract_features(file_path)
        features = features[np.newaxis, ..., np.newaxis].astype(np.float32)

        input_details = self.interpreter_audio.get_input_details()
        output_details = self.interpreter_audio.get_output_details()
        
        self.interpreter_audio.set_tensor(input_details[0]['index'], features)
        self.interpreter_audio.invoke()
        
        preds = self.interpreter_audio.get_tensor(output_details[0]['index'])
        class_idx = np.argmax(preds, axis=1)[0]
        emotion = self.le_classes[class_idx]
        confidence = preds[0][class_idx]

        return emotion, confidence
    
    def _audio_to_text(self, audio_file):
        audio, sr = librosa.load(audio_file, sr=self.SR)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        rec = KaldiRecognizer(self.vosk_model, self.SR)
        rec.SetWords(True)
        results = []
        step = 4000
        for i in range(0, len(audio_int16), step):
            chunk = audio_int16[i:i+step].tobytes()
            if rec.AcceptWaveform(chunk):
                res = json.loads(rec.Result())
                results.append(res.get("text", ""))
        res = json.loads(rec.FinalResult())
        results.append(res.get("text", ""))
        return " ".join(results).strip()
    
    def _encode_sentences(self, sentences, batch_size=32):
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                output = self.encoding_model(**encoded)
                token_embeddings = output.last_hidden_state
                attention_mask = encoded['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                sentence_embeddings = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(1)
                embeddings.append(sentence_embeddings.cpu().numpy())
        return np.vstack(embeddings)
    
    def _predict_text_prob(self, sentences):
        X_emb = self._encode_sentences(sentences).astype(np.float32)
        
        input_details = self.interpreter_text.get_input_details()
        output_details = self.interpreter_text.get_output_details()
        
        self.interpreter_text.set_tensor(input_details[0]['index'], X_emb)
        self.interpreter_text.invoke()
        
        preds = self.interpreter_text.get_tensor(output_details[0]['index'])
        return preds[0]
    
    def _one_hot_emotion(self, tone):
        vec = np.zeros(len(self.emotions), dtype=np.float32)
        if tone == "neutral" or tone == "calm":
            vec[self.emo2idx["neutral_calm"]] = 1.0
        else:
            vec[self.emo2idx[tone]] = 1.0
        return vec
    
    def _predict_severity(self, p_text, tone, eps=1e-9):
        p_text = np.array(p_text, dtype=np.float32)
        e_audio = self._one_hot_emotion(tone)

        z_t = np.log(np.clip(p_text, eps, 1.0))
        delta_z = self.M @ e_audio + self.b
        z_final = z_t + delta_z

        exp_z = np.exp(z_final - np.max(z_final))
        p_final = exp_z / np.sum(exp_z)

        S_pred = np.dot(self.w, p_final)
        return S_pred, p_final
    
    def _get_severity_class(self, p_text, tone):
        S_pred, p_final = self._predict_severity(p_text, tone)

        if S_pred <= 30:
            severity_class = "Safe"
        elif S_pred <= 50:
            severity_class = "Suspicious"
        elif S_pred <= 80:
            severity_class = "Danger"
        else:
            severity_class = "Severe"

        return S_pred, p_final, severity_class
    
    def classify_audio(self, audio_file):
        emotion, confidence = self._predict_audio(audio_file)
        print(f"Predicted emotion: {emotion}, Confidence: {confidence:.2f}\n")

        text = self._audio_to_text(audio_file)
        print(f"Predicted text: {text}\n")

        text_preds = self._predict_text_prob([text])
        print(f"INPUT: {text}")
        print(f"OUTPUT: {text_preds}\n")

        severity_pred, p_final, severity_class = self._get_severity_class(text_preds, emotion)
        print(f"text_preds = {text_preds}")
        print(f"Predicted severity: {severity_pred}")
        print(f"Predicted severity class: {severity_class}")

        return {
            "predicted_emotion": emotion,
            "emotion_confidence": float(confidence),
            "transcribed_text": text,
            "text_prediction_probabilities": text_preds.tolist(),
            "predicted_severity_score": float(severity_pred),
            "predicted_severity_class": severity_class
        }

IS_PRODUCTION = os.getenv("RENDER", False) or os.getenv("RAILWAY", False)

if IS_PRODUCTION:
    S3_URL = os.getenv("S3_URL", "")
    classifier = AudioTextClassifier(input_path="/data/models", use_s3=True, s3_url=S3_URL)
else:
    classifier = AudioTextClassifier(input_path="./models")