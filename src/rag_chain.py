import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from retriever import cargar_vector_store, recuperar_contexto

# ── Configuración ─────────────────────────────────────────────────────────────
load_dotenv()

OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """Eres un asistente médico especializado. Tu única fuente de \
información es el historial clínico del paciente que se te proporciona a \
continuación. 

Reglas estrictas:
- Responde SIEMPRE en español.
- Usa ÚNICAMENTE la información del historial proporcionado.
- Si la respuesta no está en el historial, di exactamente: \
"No encuentro esa información en el historial de este paciente."
- Sé claro, estructurado y preciso.
- No inventes datos, medicaciones ni diagnósticos.

HISTORIAL DEL PACIENTE {patient_id}:
{contexto}

PREGUNTA: {pregunta}

RESPUESTA:"""

prompt = PromptTemplate(
    input_variables=["patient_id", "contexto", "pregunta"],
    template=PROMPT_TEMPLATE,
)


# ── Inicializar LLM ───────────────────────────────────────────────────────────
def cargar_llm() -> OllamaLLM:
    """Conecta con Qwen 2.5 corriendo en Ollama."""
    return OllamaLLM(
        model    = OLLAMA_MODEL,
        base_url = OLLAMA_BASE_URL,
        temperature = 0.1,  # respuestas más deterministas, menos creativas
    )


# ── Chain principal ───────────────────────────────────────────────────────────
def construir_chain(llm, vector_store):
    """Devuelve una función que recibe patient_id + pregunta y devuelve respuesta."""

    def chain(patient_id: str, pregunta: str) -> str:
        # Paso 1: recuperar contexto relevante del paciente
        contexto = recuperar_contexto(patient_id, pregunta, vector_store)

        # Paso 2: construir el prompt con el contexto
        prompt_final = prompt.format(
            patient_id = patient_id,
            contexto   = contexto,
            pregunta   = pregunta,
        )

        # Paso 3: llamar a Qwen 2.5 y devolver la respuesta
        respuesta = llm.invoke(prompt_final)
        return respuesta.strip()

    return chain


# ── Prueba rápida ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Cargando vector store y LLM...\n")
    vs  = cargar_vector_store()
    llm = cargar_llm()
    rag = construir_chain(llm, vs)

    pruebas = [
        ("P001", "¿Qué medicación toma actualmente y para qué es cada fármaco?"),
        ("P002", "¿Cuál es la situación cardíaca actual del paciente y qué riesgos tiene?"),
        ("P003", "¿Cómo está la función renal y qué plan tiene el médico?"),
    ]

    for patient_id, pregunta in pruebas:
        print(f"{'='*55}")
        print(f"Paciente : {patient_id}")
        print(f"Pregunta : {pregunta}")
        print(f"{'='*55}")
        respuesta = rag(patient_id, pregunta)
        print(respuesta)
        print()
