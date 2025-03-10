from src.rag.db import DB
from src.rag.llm import LLMInference


class RAG:
    """
    A Retrieval-Augmented Generation (RAG) class that combines a document database
    with an LLM inference engine. It retrieves context using the DB class and uses the
    LLM to generate a response based on the query and retrieved documents.
    """

    def __init__(self,
                 collection_name: str = "med_textbooks", 
                 embedding_model_name: str = "multi-qa-MiniLM-L6-cos-v1",
                 llm_name: str="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") -> None:
        """
        Initialize the RAG instance.

        :param llm_name: The HuggingFace model name or path.
        """
        self.db = DB(collection_name=collection_name,
                     embedding_model_name=embedding_model_name)
        self.llm = LLMInference(model_name=llm_name)

    def get_response(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant documents from the DB, combine them with the query into a single prompt,
        and generate a response using the LLM.

        :param query: The user's query.
        :param top_k: Number of top relevant documents to retrieve from the database.
        :return: The generated response text.
        """
        retrieved_docs = self.db.query(query, top_k=top_k)
        context = "\n\n".join(retrieved_docs)
        
        prompt = f"""User query: {query}\nRetrieved documents: {context}\nAnalyze these documents using your internal knowledge, filtering out irrelevant information to generate the response.\nResponse:\n<think>"""
        response = self.llm.generate_response(prompt)
        return response
    

# # usage
# if __name__ == "__main__":
#     user_query = """What type of cement bonds to tooth structure, provides an anticariogenic effect, has a degree of translucency, and is non-irritating to the pulp?"""
    
#     rag = RAG()
#     response = rag.get_response(query=user_query, top_k=2)
#     print("------------------")
#     print(response)

