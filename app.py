#from dotenv import load_dotenv
#load_dotenv()

import os
import streamlit as st
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------
# 専門家定義
# --------------------------
PERSONA_SYSTEMS: Dict[str, str] = {
    "A: データサイエンティスト": (
        "あなたは熟練のデータサイエンティストです。"
        "統計学・機械学習・可視化・実験設計に明るく、"
        "専門用語を使い過ぎず、数式が必要なら簡潔に補足し、"
        "実務的なステップ（データ収集→前処理→モデル→評価→運用）で説明してください。"
        "回答は日本語。箇条書きや手順を効果的に用いてください。"
    ),
    "B: マーケティングコンサルタント": (
        "あなたは経験豊富なマーケティングコンサルタントです。"
        "STP・4P・AARRR・ペルソナ設計・CVR改善・広告運用・CRMに精通し、"
        "具体的な打ち手と測定指標(KPI)を示してください。"
        "回答は日本語。実行可能なアクションと優先順位を明示してください。"
    ),
}

DEFAULT_MODEL = "gpt-4o-mini"


def _get_api_key_from_env_or_secrets() -> str:
    # Streamlit Cloud等では st.secrets も利用可能
    key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY が見つかりません。環境変数または st.secrets に設定してください。"
        )
    return key


def _build_chain(system_message: str, model_name: str = DEFAULT_MODEL):
    # LangChain LCEL: Prompt -> LLM -> Parser
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "{user_input}"),
        ]
    )
    llm = ChatOpenAI(model=model_name, temperature=0.2)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain


# -------------------------------------------------------------
#  関数：
#   引数: 「入力テキスト」(input_text), 「ラジオボタンでの選択値」(selected_value)
#   戻り値: LLM の回答文字列
# -------------------------------------------------------------
def get_llm_answer(input_text: str, selected_value: str) -> str:
    if selected_value not in PERSONA_SYSTEMS:
        raise ValueError("未対応の選択値です。")

    # Build chain for selected persona
    system_message = PERSONA_SYSTEMS[selected_value]
    chain = _build_chain(system_message=system_message, model_name=DEFAULT_MODEL)

    # Run
    answer = chain.invoke({"user_input": input_text})
    return answer


# --------------------------
# UI
# --------------------------
st.set_page_config(page_title="LangChain Persona Demo", page_icon="💬")

st.title("💬 LangChain パーソナ切替デモ（A/B）")
st.caption("入力テキストと、専門家の種類（A/B）を選んで送信すると、選択した専門家として LLM が回答します。")

with st.expander("ℹ️ このWebアプリの概要と使い方", expanded=True):
    st.markdown(
        """
**概要**  
- 1つの入力フォームに質問やお題を入力します。  
- ラジオボタンで **A / B** いずれかの専門家を選びます。  
- 送信すると、選択した専門家の「システムメッセージ（役割指示）」を使って LLM に問い合わせ、結果を表示します。  

**操作方法**  
1. 下の入力欄に質問・課題・文章などを記入  
2. 専門家（AまたはB）を選択  
3. **送信** を押す → 画面下部に回答が表示されます  

        """
    )

# Persona selection
selected = st.radio(
    "専門家の種類を選択：",
    list(PERSONA_SYSTEMS.keys()),
    horizontal=True
)

# Input area
user_text = st.text_area("入力テキスト", value="", height=150, placeholder="ここに質問や文章を入力してください…")

col1, col2 = st.columns([1, 3])
with col1:
    send = st.button("送信", type="primary")

# Optional: model override in sidebar
with st.sidebar:
    st.header("設定")
    st.write("OpenAI の API キーは環境変数または `st.secrets` で設定してください。")
    st.write(f"モデル: `{DEFAULT_MODEL}`（コードで変更可）")

# Handle submit
if send:
    if not user_text.strip():
        st.warning("入力テキストを記入してください。")
    else:
        try:
            _ = _get_api_key_from_env_or_secrets()  # 早期検証
            with st.spinner("LLM に問い合わせ中…"):
                output = get_llm_answer(user_text, selected)
            st.success("回答を取得しました。")
            st.markdown("### 🧠 回答")
            st.write(output)
        except Exception as e:
            st.error(f"エラー: {e}")
