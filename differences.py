import polars as pl
import langchain_openai as lco
import langchain_core as lc
from langchain_community.embeddings import OllamaEmbeddings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from itertools import chain

label_encoder = joblib.load("Models/label_encoder.joblib")
svm_classifier = joblib.load("Models/svm_model.joblib")
embeddings = OllamaEmbeddings(base_url="http://192.168.1.200:11434", model="hf.co/bartowski/granite-embedding-278m-multilingual-GGUF:IQ3_M")

if __name__ == "__main__":
    label_encoder = joblib.load("Models/label_encoder.joblib")
    svm_classifier = joblib.load("Models/svm_model.joblib")
    embeddings = OllamaEmbeddings(base_url="http://192.168.1.200:11434", model="hf.co/bartowski/granite-embedding-278m-multilingual-GGUF:IQ3_M")

    texts = pl.read_csv("Data/requirements.csv")
    all_texts = texts["Requirement Text"].to_list()

    batch_size = 32  # 適切なバッチサイズを設定 (モデルや環境に合わせて調整)
    text_embeddings = []

    # tqdmで全テキストをバッチに分けて処理し、進捗を表示
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Embedding Texts"):
        # バッチを取得
        batch = all_texts[i:i + batch_size]
    
        # 埋め込みを計算
        batch_embeddings = embeddings.embed_documents(batch)
    
        # 結果の埋め込みをリストに追加
        text_embeddings.extend(batch_embeddings)

    X = np.array(text_embeddings)

    y_pred = svm_classifier.predict(X)
    decision_values = svm_classifier.decision_function(X)

    threshold = 0.1

    samples_near_boundary = []
    for i, distances in enumerate(decision_values):
        if np.any(np.abs(distances) < threshold):
            samples_near_boundary.append({
                "Requirement Text": all_texts[i],
                "Predicted Class": label_encoder.inverse_transform([y_pred[i]])[0],
                "Distances": distances
            })
    
    if samples_near_boundary:
        df_near_boundary = pl.DataFrame(samples_near_boundary)
        df_near_boundary.write_csv("Data/exported_samples_near_boundary.csv")
        print(f"Samples near decision boundary saved to exported_samples_near_boundary.csv")

