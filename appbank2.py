import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import os
import warnings

warnings.filterwarnings("ignore")


# CONFIG DA PÁGINA

st.set_page_config(page_title="Análise Bancária", page_icon="💰", layout="wide")
sns.set_theme(style="whitegrid")

st.sidebar.title("Configurações")


# AUTENTICAÇÃO SIMPLES

if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False

password = st.sidebar.text_input("Senha de acesso", type="password", help="Senha de exemplo: ebac2025")
if st.sidebar.button("Entrar") and not st.session_state["auth_ok"]:
    if password == "ebac2025":
        st.session_state["auth_ok"] = True
        st.sidebar.success("Acesso liberado!")
    else:
        st.sidebar.error("Senha incorreta.")

if not st.session_state["auth_ok"]:
    st.warning("🔒 Digite a senha **ebac2025** na barra lateral para acessar o dashboard.")
    st.stop()


# PALETA DE CORES

palette_option = st.sidebar.selectbox(
    "Paleta de cores",
    ["Padrão", "Azul", "Verde", "Rosa"]
)
if palette_option == "Padrão":
    sns.set_palette("deep")
elif palette_option == "Azul":
    sns.set_palette("Blues")
elif palette_option == "Verde":
    sns.set_palette("Greens")
else:
    sns.set_palette("PuRd")


# UPLOAD OPCIONAL

uploaded_file = st.sidebar.file_uploader("📂 Enviar CSV (opcional)", type=["csv"])


# FUNÇÃO INTELIGENTE DE CARREGAMENTO

@st.cache_data(show_spinner=True)
def carregar_dados(upload):
    """
    Prioridade:
    1) CSV enviado pelo usuário
    2) bank-additional-full.csv na pasta principal
    3) data/input/bank-additional-full.csv
    4) data/input/bank-additional-full-40.csv
    """
    caminhos = [
        "bank-additional-full.csv",
        os.path.join("data", "input", "bank-additional-full.csv"),
        os.path.join("data", "input", "bank-additional-full-40.csv"),
    ]

    # 1) Upload
    if upload is not None:
        df = pd.read_csv(upload, sep=";")
        return df, "Arquivo enviado pelo usuário"

    # 2) Caminhos locais
    for path in caminhos:
        if os.path.exists(path):
            df = pd.read_csv(path, sep=";")
            return df, f"Arquivo local: {path}"

    # 3) Nada encontrado
    return None, None

df, origem = carregar_dados(uploaded_file)

if df is None:
    st.error(
        "❌ Não encontrei nenhum arquivo de dados.\n\n"
        "Faça uma das opções:\n"
        "- Coloque **bank-additional-full.csv** na mesma pasta do `appbank2.py`, ou\n"
        "- Coloque em `data/input/`, ou\n"
        "- Envie um CSV pela barra lateral."
    )
    st.stop()


# CABEÇALHO

st.markdown(
    """
    <h1 style='text-align:center;'>💰 Análise Bancária</h1>
    <p style='text-align:center;color:gray;'>Dashboard profissional de campanhas de marketing bancário</p>
    """,
    unsafe_allow_html=True,
)
st.markdown(f"**Origem dos dados:** {origem}")
st.markdown("---")


# FILTROS GLOBAIS (SIDEBAR)

df_filtered = df.copy()

if "age" in df_filtered.columns:
    min_age = int(df_filtered["age"].min())
    max_age = int(df_filtered["age"].max())
    faixa = st.sidebar.slider("Idade (filtro global)", min_age, max_age, (min_age, max_age))
    df_filtered = df_filtered[(df_filtered["age"] >= faixa[0]) & (df_filtered["age"] <= faixa[1])]

if "job" in df_filtered.columns:
    jobs = sorted(df["job"].dropna().unique().tolist())
    jobs.insert(0, "Todas")
    sel_jobs = st.sidebar.multiselect("Profissões (filtro global)", jobs, default=["Todas"])
    if "Todas" not in sel_jobs:
        df_filtered = df_filtered[df_filtered["job"].isin(sel_jobs)]

st.sidebar.success(f"Registros após filtros: {df_filtered.shape[0]}")

tem_y = "y" in df.columns


# TABS

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Visão Geral", "📈 Análises", "🎛️ Escalonamento", "📥 Exportação", "🤖 Modelo"]
)

#  TAB 1 - VISÃO GERAL 
with tab1:
    st.subheader("Estrutura da Base")
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros (original)", df.shape[0])
    c2.metric("Registros filtrados", df_filtered.shape[0])
    c3.metric("Número de variáveis", df.shape[1])

    st.write("### Amostra dos dados filtrados")
    st.dataframe(df_filtered.head(), use_container_width=True)

    if tem_y:
        st.write("### Distribuição da variável alvo `y`")
        y_counts = df_filtered["y"].value_counts(normalize=True) * 100
        y_df = y_counts.sort_index().reset_index()
        y_df.columns = ["y", "percentual"]
        fig, ax = plt.subplots()
        sns.barplot(data=y_df, x="y", y="percentual", ax=ax)
        ax.bar_label(ax.containers[0])
        ax.set_ylabel("%")
        st.pyplot(fig)

#  TAB 2 - ANALISES 
with tab2:
    st.subheader("Análises Exploratórias")

    if "age" in df_filtered.columns:
        st.write("### Distribuição de Idade")
        fig, ax = plt.subplots()
        sns.histplot(df_filtered["age"], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    if "duration" in df_filtered.columns:
        st.write("### Distribuição da Duração da Ligação (duration)")
        fig, ax = plt.subplots()
        sns.histplot(df_filtered["duration"], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    if "campaign" in df_filtered.columns and tem_y:
        st.write("### Relação campaign x y")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df_filtered, x="y", y="campaign", ax=ax)
        st.pyplot(fig)

# TAB 3 - ESCALONAMENTO 
with tab3:
    st.subheader("Escalonamento de Variáveis Numéricas")

    num_cols = df_filtered.select_dtypes(include=["int64", "float64"]).columns.tolist()
    st.write("Colunas numéricas detectadas:", num_cols)

    metodo = st.radio("Método de Escalonamento", ["Nenhum", "StandardScaler", "MinMaxScaler"])

    if metodo != "Nenhum" and num_cols:
        df_scale = df_filtered.copy()
        scaler = StandardScaler() if metodo == "StandardScaler" else MinMaxScaler()
        df_scale[num_cols] = scaler.fit_transform(df_scale[num_cols])

        c1, c2 = st.columns(2)
        with c1:
            st.write("Antes do escalonamento:")
            st.dataframe(df_filtered[num_cols].describe().T)
        with c2:
            st.write("Depois do escalonamento:")
            st.dataframe(df_scale[num_cols].describe().T)
    else:
        st.info("Escolha um método de escalonamento para ver o efeito.")

#  TAB 4 - EXPORTAÇÃO 
with tab4:
    st.subheader("Exportação dos Dados Filtrados")

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_filtered.to_excel(writer, index=False, sheet_name="Dados Filtrados")

    st.download_button(
        "📥 Baixar Excel",
        data=buffer.getvalue(),
        file_name="analise_bancaria_filtrada.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# FUNÇÃO DE TREINO DO MODELO

@st.cache_data(show_spinner=True)
def treinar_random_forest(df_train, target_col):
    df_model = df_train.dropna(subset=[target_col]).copy()
    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    # Mapeia texto binário para 0/1 se necessário
    if y.dtype == "O":
        mapa = {"yes": 1, "no": 0, "sim": 1, "não": 0, "nao": 0}
        y = y.str.lower().map(mapa)

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    pipe = Pipeline(steps=[("prep", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y if y.nunique() == 2 else None,
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None

    cm = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred)

    return pipe, acc, auc, cm, rep

#TAB 5 - MODELO 
with tab5:
    st.subheader("Modelo de Machine Learning (RandomForest)")

    if not tem_y:
        st.error("A coluna `y` não existe na base. Não é possível treinar o modelo.")
    else:
        modelo, acc, auc, cm, rep = treinar_random_forest(df_filtered, "y")

        c1, c2 = st.columns(2)
        c1.metric("Acurácia", f"{acc:.3f}")
        if auc is not None:
            c2.metric("AUC ROC", f"{auc:.3f}")

        st.write("### Relatório de Classificação")
        st.text(rep)

        st.write("### Matriz de Confusão")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>Dashboard desenvolvido com ❤️ por Alessandra</p>",
    unsafe_allow_html=True,
)

