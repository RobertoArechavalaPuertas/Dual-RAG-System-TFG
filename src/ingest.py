import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ── Cargar configuración desde .env ──────────────────────────────────────────
load_dotenv()

PACIENTES_PATH = Path(os.getenv("PACIENTES_PATH", "./datos/pacientes"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 50))

# Modelo de embeddings local (se descarga automáticamente la primera vez, ~90 MB)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ── 1. LOADER — extrae texto de un PDF ───────────────────────────────────────
def cargar_pdf(ruta_pdf: Path) -> str:
    """Abre un PDF y devuelve todo su texto como string."""
    texto = ""
    with fitz.open(ruta_pdf) as doc:
        for pagina in doc:
            texto += pagina.get_text()
    return texto


def extraer_patient_id(nombre_archivo: str) -> str:
    """Extrae el patient_id del nombre del archivo.
    
    Ejemplo: 'P001_Carmen_Vidal_Soler.pdf' → 'P001'
    """
    return nombre_archivo.split("_")[0]


# ── 2. SPLITTER — trocea el texto en chunks ───────────────────────────────────
def trocear_texto(texto: str) -> list[str]:
    """Divide el texto en chunks con solapamiento."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(texto)


# ── 3. EMBEDDINGS + VECTOR STORE — indexa en ChromaDB ────────────────────────
def indexar_pacientes():
    """Proceso completo: lee todos los PDFs y los indexa en ChromaDB."""

    # Verificar que la carpeta de pacientes existe y tiene PDFs
    if not PACIENTES_PATH.exists():
        print(f"ERROR: No se encontró la carpeta {PACIENTES_PATH}")
        sys.exit(1)

    pdfs = list(PACIENTES_PATH.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: No hay PDFs en {PACIENTES_PATH}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  Iniciando indexación de {len(pdfs)} pacientes")
    print(f"{'='*50}\n")

    # Preparar listas para inserción en bloque en ChromaDB
    todos_los_textos    = []
    todos_los_metadatos = []

    for ruta_pdf in sorted(pdfs):
        nombre    = ruta_pdf.name
        patient_id = extraer_patient_id(nombre)

        print(f"[{patient_id}] Procesando: {nombre}")

        # Paso 1 — Loader
        texto = cargar_pdf(ruta_pdf)
        print(f"  → Texto extraído: {len(texto):,} caracteres")

        # Paso 2 — Splitter
        chunks = trocear_texto(texto)
        print(f"  → Chunks generados: {len(chunks)}")

        # Preparar textos y metadatos para este paciente
        for i, chunk in enumerate(chunks):
            todos_los_textos.append(chunk)
            todos_los_metadatos.append({
                "patient_id":    patient_id,
                "nombre_archivo": nombre,
                "chunk_index":   i,
                "total_chunks":  len(chunks),
            })

        print(f"  ✓ {patient_id} listo\n")

    # Paso 3 — Embeddings + inserción en ChromaDB
    print("Cargando modelo de embeddings...")
    print("(La primera vez descarga ~90 MB, ten paciencia)\n")

    embedding_fn = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    print("Indexando todos los chunks en ChromaDB...")

    vector_store = Chroma.from_texts(
        texts     = todos_los_textos,
        metadatas = todos_los_metadatos,
        embedding = embedding_fn,
        persist_directory = CHROMA_DB_PATH,
        collection_name   = "historiales_clinicos",
    )

    print(f"\n{'='*50}")
    print(f"  ✓ Indexación completada")
    print(f"  Total chunks indexados: {len(todos_los_textos)}")
    print(f"  Base de datos guardada en: {CHROMA_DB_PATH}")
    print(f"{'='*50}\n")

    return vector_store


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    indexar_pacientes()
