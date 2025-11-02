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

label_encoder = joblib.load("Models/label_encoder.joblib")
svm_classifier = joblib.load("Models/svm_model.joblib")
embeddings = OllamaEmbeddings(base_url="http://192.168.1.200:11434", model="hf.co/bartowski/granite-embedding-278m-multilingual-GGUF:IQ3_M")

def classify_requirement(requirement_text):
    # 埋め込みを計算
    req_embedding = embeddings.embed_query(requirement_text)
    req_embedding_np = np.array(req_embedding).reshape(1, -1)
    
    # 分類を実行
    predicted_label_encoded = svm_classifier.predict(req_embedding_np)
    
    # エンコードされたラベルを元のラベルに戻す
    predicted_label = label_encoder.inverse_transform(predicted_label_encoded)
    
    return predicted_label[0]

if __name__ == "__main__":
    test_requirement = "The system shall allow users to reset their passwords via email."
    predicted_class = classify_requirement(test_requirement)
    print(f"Predicted class for the requirement: {predicted_class}")
