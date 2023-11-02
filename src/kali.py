# encoding: utf-8

import logging
import os
import openai

from llama_index import GPTVectorStoreIndex, Document
from llama_index.vector_stores import DocArrayInMemoryVectorStore
from llama_index.storage.storage_context import StorageContext


if os.getenv("OPENAI_URL"):
    openai.api_base = os.getenv("OPENAI_URL")
    openai.verify_ssl_certs = False

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from kali_data import kali

logger = logging.getLogger()

# handler = logging.StreamHandler(stream=sys.stdout)
# handler.setFormatter(LlamaIndexFormatter())
# logger.addHandler(handler)


# build a memory store for kali data
def get_conventions_collectives_query_engine():
    VECTOR_STORE_BACKUP_PATH = "/tmp/kali"

    vector_store_backup_exist = os.path.exists(VECTOR_STORE_BACKUP_PATH)

    vector_store = DocArrayInMemoryVectorStore(index_path=VECTOR_STORE_BACKUP_PATH)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if not vector_store_backup_exist:
        documents = []
        for cc in kali:
            # todo: ajouter des métiers dans les données
            doc = Document(
                text=f"IDCC{cc.get('idcc','')} : {cc.get('title','')}, {cc.get('shortTile', '')}"
            )
            doc.metadata = {
                "idcc": cc.get("idcc", ""),
                "title": cc.get("shortTile", ""),
            }
            documents.append(doc)

        index = GPTVectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=True
        )
        storage_context.persist()
    else:
        index = GPTVectorStoreIndex.from_vector_store(vector_store=vector_store)

    query_engine = index.as_query_engine()

    return query_engine


convention_collective_query_engine = get_conventions_collectives_query_engine()


# def get_convention_collective(query: str):
#     response = convention_collective_query_engine.query(
#         f'Quelles sont les conventions collectives liées à "{query}" ? Renvoies un format CSV avec les metadonnées dans des colonnes'
#     )
#     results = re.findall(r"(\d+),(.*)", str(response))
#     if len(results) == 1:
#         return f"RESULT=IDCC{results[0][0]}"
#     elif len(results) > 1:
#         return "\n -" + "\n -".join(map(lambda a: f" - IDCC{a[0]}: {a[1]}", results))
#     return None
