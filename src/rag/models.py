"""Model clients, created once and reused (the course app's get_llm/get_embeddings pattern)."""

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag import config


@lru_cache(maxsize=None)
def get_llm(model: str = config.CHAT_MODEL) -> ChatOpenAI:
    # max_retries: the OpenAI client backs off exponentially on 429s (gpt-4.1 is limited to 30K TPM
    # on this account), so parallel judge calls wait instead of failing.
    return ChatOpenAI(model=model, temperature=config.TEMPERATURE, max_retries=10)


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
