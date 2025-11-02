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
# 💡 tqdmをインポート
from tqdm import tqdm
# 埋め込みリストを効率的に結合するために chain をインポート
from itertools import chain

texts = pl.read_csv("Software-Requirements-Classification/PROMISE.csv")

embeddings = OllamaEmbeddings(base_url="http://192.168.1.200:11434", model="hf.co/bartowski/granite-embedding-278m-multilingual-GGUF:IQ3_M")

# --- 埋め込み処理をtqdmでラップする ---
all_texts = texts["RequirementText"].to_list()
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

y = texts["_class_"].to_list()

le = LabelEncoder()

y_encoded = le.fit_transform(y)

# X (NumPy配列) と y_encoded (NumPy配列) を使って train_test_split を実行
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

target_names = le.classes_

svm_classifier = SVC(kernel='linear', random_state=42, class_weight='balanced')
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
report = classification_report(y_test, y_pred, target_names=target_names)
print(report)

print("Classes:", target_names)
print("Label mapping:", dict(zip(target_names, range(len(target_names)))))
print("Number of classes:", len(target_names))

print("Saving the LabelEncoder to 'label_encoder.joblib'...")
joblib.dump(le, "Models/label_encoder.joblib")

print("Saving the trained SVM model to 'svm_model.joblib'...")
joblib.dump(svm_classifier, "Models/svm_model.joblib")
