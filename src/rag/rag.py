from src.rag.db import DB
from src.rag.llm import LLMInference
from typing import List


class RAG:
    """
    A Retrieval-Augmented Generation (RAG) class that combines a document database
    with an LLM inference engine. It retrieves context using the DB class and uses the
    LLM to generate a response based on the query and retrieved documents.
    """

    def __init__(self,
                 collection_name: str = "medical_rag",
                 embedding_model_name: str = "multi-qa-MiniLM-L6-cos-v1",
                 llm_name: str="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") -> None:
        """
        Initialize the RAG instance.

        :param llm_name: The HuggingFace model name or path.
        """
        self.db = DB(collection_name=collection_name,
                     embedding_model_name=embedding_model_name)
        self.llm = LLMInference(model_name=llm_name)

    def get_response(self, queries: List[str], top_k: int = 3) -> List[str]:
        """
        Retrieve relevant documents, construct prompts, and generate responses for a batch of queries.

        :param queries: A list of user queries.
        :param top_k: Number of top relevant documents to retrieve for each query.
        :return: A list of generated response texts corresponding to each input query.
        """
        if not queries:
            return []

        print(f"Processing batch of {len(queries)} queries...")
        batch_retrieved_docs = self.db.query(queries, top_k=top_k)

        final_prompts = []
        for i, query in enumerate(queries):
            retrieved_docs = batch_retrieved_docs[i]
            if retrieved_docs:
                context = "\n\n".join(retrieved_docs)
                prompt = f"""Based on the following retrieved documents, answer the user's query. Filter out irrelevant information and synthesize the answer.

Retrieved documents:
---
{context}
---

User query: {query}

Answer:"""
            else:
                print(f"Warning: No documents retrieved for query: {query[:50]}...")
                prompt = f"""Answer the following user query based on your internal knowledge.

User query: {query}

Answer:"""
            final_prompts.append(prompt)
            
        responses = self.llm.generate(final_prompts)
        return responses
    

# # usage
# if __name__ == "__main__":
#     user_query = """What type of cement bonds to tooth structure, provides an anticariogenic effect, has a degree of translucency, and is non-irritating to the pulp?"""
    
#     rag = RAG()
#     response = rag.get_response(queries=[user_query], top_k=2)
#     print("------------------")
#     print(response)
