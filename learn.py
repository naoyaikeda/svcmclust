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

# Store original texts to relate back after splitting
original_texts = texts["RequirementText"].to_numpy()

# X (NumPy配列) と y_encoded (NumPy配列) を使って train_test_split を実行
X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
    X, y_encoded, original_texts, test_size=0.2, random_state=42, stratify=y_encoded
)

target_names = le.classes_

svm_classifier = SVC(kernel='linear', random_state=42, class_weight='balanced')
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
report = classification_report(y_test, y_pred, target_names=target_names)
print(report)

# --- 識別境界に近いサンプルを特定してCSVに出力 ---
# decision_functionで各サンプルと識別境界の距離を計算
decision_values = svm_classifier.decision_function(X_test)

# 識別境界に近いサンプルを抽出（絶対値が0.1未満のサンプル）
threshold = 0.1

# Handle multiclass vs. binary case for decision_values
if len(decision_values.shape) > 1:
    # For multiclass, the distance to the boundary can be estimated by the difference
    # between the two highest decision function values.
    sorted_decision_values = np.sort(decision_values, axis=1)
    distances = sorted_decision_values[:, -1] - sorted_decision_values[:, -2]
else:
    # For binary, it's just the absolute value of the decision function
    distances = np.abs(decision_values)

near_boundary_indices = np.where(distances < threshold)[0]


# 該当するサンプルの情報を収集
boundary_samples = []
for idx in near_boundary_indices:
    boundary_samples.append({
        "RequirementText": texts_test[idx],
        "TrueLabel": le.inverse_transform([y_test[idx]])[0],
        "PredictedLabel": le.inverse_transform([y_pred[idx]])[0],
        "Distance": distances[idx]
    })

# Polars DataFrameに変換してCSVに出力
if boundary_samples:
    df_boundary = pl.DataFrame(boundary_samples)
    df_boundary.write_csv("Data/samples_near_boundary.csv")
    print(f"\nFound {len(df_boundary)} samples near the decision boundary. Saved to 'samples_near_boundary.csv'.")
else:
    print("\nNo samples found near the decision boundary with the current threshold.")


print("Classes:", target_names)
print("Label mapping:", dict(zip(target_names, range(len(target_names)))))
print("Number of classes:", len(target_names))

print("Saving the LabelEncoder to 'label_encoder.joblib'...")
joblib.dump(le, "Models/label_encoder.joblib")

print("Saving the trained SVM model to 'svm_model.joblib'...")
joblib.dump(svm_classifier, "Models/svm_model.joblib")
