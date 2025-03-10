from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from typing import List, Any
from datasets import load_dataset
import time


class DB:
    def __init__(self,
                 collection_name: str = "med_textbooks",
                 embedding_model_name: str = "multi-qa-MiniLM-L6-cos-v1") -> None:
        """
        Initialize the DB instance.

        :param db_path: Directory path where the database is stored or will be created.
        :param embedding_model_name: Name of the embedding model used for generating embeddings.
        """
        self.collection_name = collection_name
        self.db_path = os.path.join(os.getcwd(), collection_name)
        self.embedding_model_name = embedding_model_name
        self.db = None

        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}
        self.embedder = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        if os.path.exists(self.db_path):
            # print("DB exists, loading it...")
            self.db = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedder,
                persist_directory=self.db_path,
            )
            # print("DB loaded.")
        else:
            # print("DB does not exist, creating it...")
            os.makedirs(self.db_path, exist_ok=True)
            # start_time = time.time()
            self._populate_db()
            # end_time = time.time()
            # print("DB populated.")
            # print(f"Time taken to populate DB: {end_time - start_time} sec")

    def _populate_db(self) -> None:
        """
        Populate the database from the dataset.
        Loads the 'MedRAG/textbooks' dataset and creates a new Chroma database from the 'contents' field.
        """
        ds = load_dataset("MedRAG/textbooks")
        contents = ds["train"]["contents"]
        self.db = Chroma.from_texts(
            texts=contents,
            embedding=self.embedder,
            persist_directory=self.db_path,
            collection_name=self.collection_name,
        )

    def query(self, query: str, top_k: int = 3) -> List[Any]:
        """
        Query the database for the top-k most relevant chunks based on the input query.

        :param query: The user's search query.
        :param top_k: The number of top relevant results to retrieve.
        :return: A list of search results.
        """
        results = self.db.similarity_search(query, k=top_k)
        retrieved_docs = [doc.page_content for doc in results]
        return retrieved_docs

    def close(self) -> None:
        """
        Close the database connection.
        """
        if self.db:
            self.db.close()
        

# # usage
# if __name__ == "__main__":
#     user_query = """What type of cement bonds to tooth structure, provides an anticariogenic effect, 
#                 has a degree of translucency, and is non-irritating to the pulp?"""
    
#     db = DB()
#     relevant_docs = db.query(query=user_query, top_k=2)
#     print("------------------")
#     print(relevant_docs)