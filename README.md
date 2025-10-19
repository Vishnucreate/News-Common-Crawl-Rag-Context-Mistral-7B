# News-Common-Crawl-Rag-Context-Mistral-7B
Tried implementing a Rag application on the open source dataset Crawl with mistal-7B 
# Agentic RAG with Mistral-7B and C4 Dataset  
This notebook builds a Retrieval-Augmented Generation (RAG) pipeline using the open-source model **mistralai/Mistral-7B-Instruct-v0.2** in 4-bit quantized mode for Colab A100 GPUs. It retrieves relevant text from the large `allenai/c4` “realnewslike” dataset, encodes them with `all-MiniLM-L6-v2`, stores embeddings in FAISS, and generates grounded answers.  

Install dependencies:  
```bash
!pip install -q transformers sentence-transformers faiss-cpu datasets accelerate bitsandbytes
Load model:

from transformers import pipeline
pipe = pipeline("text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype="auto",
    device_map="auto",
    load_in_4bit=True)


Load dataset (~350k English news-style docs):

from datasets import load_dataset
ds = load_dataset("allenai/c4","realnewslike",split="train[:1%]")
docs=[d["text"] for d in ds if d["text"]]


Build embeddings and FAISS index:

from sentence_transformers import SentenceTransformer; import numpy as np,faiss,gc
embedder=SentenceTransformer("all-MiniLM-L6-v2")
embeddings=embedder.encode(docs,batch_size=64,show_progress_bar=True,convert_to_numpy=True)
index=faiss.IndexFlatL2(embeddings.shape[1]); index.add(embeddings); gc.collect()


Define retrieval and RAG functions:

def retrieve(q,k=10):
    q_emb=embedder.encode([q],convert_to_numpy=True)
    D,I=index.search(q_emb,k)
    return [docs[i] for i in I[0]]

def agentic_rag(q,top_k=10):
    ctx="\n\n".join(retrieve(q,top_k))
    prompt=f"You are a helpful assistant.\n\nContext:\n{ctx}\n\nQuestion:{q}\n\nAnswer in detail:"
    out=pipe(prompt,max_new_tokens=512,temperature=0.7,top_p=0.9,repetition_penalty=1.1,do_sample=True,pad_token_id=pipe.tokenizer.eos_token_id)
    return out[0]["generated_text"].split('Answer in detail:')[-1].strip()


Run a test query:

print(agentic_rag("Explain a quantization technique better than LoRA."))


Performance (Colab Pro A100): GPU ≈ 8–10 GB VRAM, System RAM ≈ 60–80 GB, Disk ≈ 10 GB, Load ≈ 15 min for 1 % slice.

Credits:
Model – mistralai/Mistral-7B-Instruct-v0.2

Dataset – AllenAI C4 realnewslike

Embeddings – Sentence-Transformers MiniLM L6-v2

Author – Vishnu Sunil
