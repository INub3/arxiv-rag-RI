import os
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymilvus import connections, Collection
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import time
import random

# =========================================================
# Configuración de página
# =========================================================
st.set_page_config(page_title="RAG arXiv Chat", page_icon="📚", layout="wide")

# =========================================================
# Carga de recursos (cacheada para no recargar en cada mensaje)
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-base-en-v1.5")

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def load_collection():
    connections.connect(alias="default", uri="./arxiv.db")
    collection = Collection("papers")
    collection.load()
    return collection

@st.cache_resource
def load_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


embedding_model = load_embedding_model()
reranker = load_reranker()
collection = load_collection()
gemini_client = load_gemini_client()

GEMINI_MODEL = "gemini-3.5-flash"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RAG_SYSTEM_INSTRUCTION = """Eres un asistente experto en literatura científica que responde preguntas
basándote ÚNICAMENTE en los fragmentos de evidencia proporcionados (abstracts de papers de arXiv).

Reglas estrictas:
1. Responde solo con información contenida en las evidencias.
2. Si las evidencias no contienen información suficiente para responder la consulta,
   indícalo explícitamente diciendo algo como: "El corpus no contiene información suficiente
   para responder esta consulta con certeza."
3. Si integras información de varios documentos, sé explícito al respecto.
4. No inventes datos, autores, ni resultados que no estén en las evidencias.
5. Responde en inglés si la consulta está en inglés, o en español si la consulta está en español.
"""

# =========================================================
# Funciones del pipeline (idénticas a las del notebook)
# =========================================================
def retrieve(query, top_k=20):
    query_with_prefix = BGE_QUERY_PREFIX + query
    query_embedding = embedding_model.encode(query_with_prefix, normalize_embeddings=True).tolist()

    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )

    st.write("DEBUG num_entities:", collection.num_entities)
    st.write("DEBUG resultados encontrados:", len(results[0]))

    return [{"text": hit.entity.get("text"), "score": float(hit.score)} for hit in results[0]]


def rerank(query, candidates, top_k=5):
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    reranked = [{"text": c["text"], "rerank_score": float(s)} for c, s in zip(candidates, scores)]
    return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


def search(query, top_k_retrieve=20, top_k_final=5):
    candidates = retrieve(query, top_k=top_k_retrieve)
    return rerank(query, candidates, top_k=top_k_final)


def build_prompt(query, evidences):
    context_blocks = [f"[Evidence {i}]\n{ev['text']}" for i, ev in enumerate(evidences, start=1)]
    context_text = "\n\n".join(context_blocks)
    return f"""Consulta del usuario: {query}

Evidencias recuperadas del corpus:

{context_text}

Instrucción: Responde la consulta del usuario basándote únicamente en las evidencias anteriores,
siguiendo las reglas del sistema."""


def generate_answer(query, evidences, temperature=0.2, max_retries=4):
    prompt = build_prompt(query, evidences)

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=RAG_SYSTEM_INSTRUCTION,
                    temperature=temperature,
                    max_output_tokens=800,
                ),
            )
            return response.text
        except genai_errors.ServerError:
            wait = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait)

    return "No fue posible generar una respuesta en este momento debido a alta demanda del servicio."


def rag_pipeline(query, top_k_retrieve=20, top_k_final=5):
    evidences = search(query, top_k_retrieve=top_k_retrieve, top_k_final=top_k_final)
    answer = generate_answer(query, evidences)
    return {"query": query, "answer": answer, "evidences": evidences}


# =========================================================
# Interfaz de chat
# =========================================================

st.title("📚 RAG sobre arXiv Paper Abstracts")
st.caption("Sistema de Recuperación de Información Aumentada por Generación (embeddings + re-ranking + Gemini)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "evidences" in msg:
            with st.expander("📄 Ver evidencias utilizadas"):
                for i, ev in enumerate(msg["evidences"], start=1):
                    st.markdown(f"**Evidencia {i}** — Score: `{ev['rerank_score']:.4f}`")
                    st.write(ev["text"][:500] + ("..." if len(ev["text"]) > 500 else ""))
                    st.divider()

# Input del usuario
user_query = st.chat_input("Escribe tu consulta sobre papers de arXiv...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Buscando evidencias y generando respuesta..."):
            result = rag_pipeline(user_query)
        st.markdown(result["answer"])
        with st.expander("📄 Ver evidencias utilizadas"):
            for i, ev in enumerate(result["evidences"], start=1):
                st.markdown(f"**Evidencia {i}** — Score: `{ev['rerank_score']:.4f}`")
                st.write(ev["text"][:500] + ("..." if len(ev["text"]) > 500 else ""))
                st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "evidences": result["evidences"]
    })
