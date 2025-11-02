# SVCMCLUST

埋め込みベクトルを用いてSVMでソフトウェア要求仕様を分類する実験的プロジェクトです。

## 概要

このプロジェクトは、[LangChain](https://www.langchain.com/) を使用してテキストの埋め込みベクトルを生成し、[Scikit-learn](https://scikit-learn.org/stable/) のサポートベクターマシン（SVM）を用いて、ソフトウェア要求仕様を分類します。

データセットとして `Software-Requirements-Classification/PROMISE.csv` を使用しています。

## 技術スタック

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (推奨パッケージマネージャー)
- [LangChain](https://www.langchain.com/): 埋め込み生成
- [Scikit-learn](https://scikit-learn.org/stable/): SVM分類器
- [Polars](https://pola.rs/): データ操作
- [Joblib](https://joblib.readthedocs.io/): モデルのシリアライズ
- [tqdm](https://tqdm.github.io/): プログレスバー表示

## ファイル構成

```
.
├── .gitignore
├── .gitmodules
├── .python-version
├── GEMINI.md
├── inference.py      # 学習済みモデルで要求仕様を分類するスクリプト
├── learn.py          # 要求仕様データセットでSVMモデルを学習するスクリプト
├── pyproject.toml    # プロジェクト設定と依存関係
├── README.md         # このファイル
├── uv.lock
├── Models/           # 学習済みモデルとラベルエンコーダーを保存
│   └── .gitignore
└── Software-Requirements-Classification/ # データセット (Gitサブモジュール)
    ├── ...
    └── PROMISE.csv
```

## セットアップ

1.  **リポジトリのクローンとサブモジュールの初期化:**

    ```bash
    git clone --recurse-submodules https://github.com/your-username/svcmclust.git
    cd svcmclust
    ```

2.  **Python環境のセットアップと依存関係のインストール:**

    [uv](https://github.com/astral-sh/uv) の使用を推奨します。

    ```bash
    uv venv
    source .venv/bin/activate  # Windowsの場合は .venv\Scripts\activate
    uv pip install -e .
    ```

3. **Ollamaのセットアップ**
   このプロジェクトはOllamaを使用して埋め込みを生成します。Ollamaをインストールし、必要なモデルをプルしてください。
   ```bash
   ollama pull hf.co/bartowski/granite-embedding-278m-multilingual-GGUF:IQ3_M
   ```
   また、`learn.py`と`inference.py`内のOllamaのベースURL (`base_url`) をご自身の環境に合わせて変更してください。

## 使用方法

1.  **モデルの学習:**

    `learn.py` を実行して、`PROMISE.csv` データセットからSVMモデルを学習し、`Models/` ディレクトリに保存します。

    ```bash
    python learn.py
    ```

2.  **要求仕様の分類:**

    `inference.py` を実行して、サンプルテキストの分類を試すことができます。

    ```bash
    python inference.py
    ```

    `inference.py` 内の `test_requirement` 変数を変更することで、任意のテキストを分類できます。