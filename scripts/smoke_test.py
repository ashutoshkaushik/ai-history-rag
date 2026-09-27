"""Phase 0 check: one chat call and one embedding call through LangChain."""

import time

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag import config

t = time.perf_counter()
reply = ChatOpenAI(model=config.CHAT_MODEL, temperature=0).invoke(
    "In one sentence: who wrote 'Computing Machinery and Intelligence'?"
)
print(f"[chat]  {config.CHAT_MODEL}: {reply.content}  ({time.perf_counter() - t:.2f}s)")
print(f"        tokens: {reply.usage_metadata}")

t = time.perf_counter()
vec = OpenAIEmbeddings(model=config.EMBEDDING_MODEL).embed_query("Dartmouth summer research project")
print(f"[embed] {config.EMBEDDING_MODEL}: {len(vec)} dims  ({time.perf_counter() - t:.2f}s)")
