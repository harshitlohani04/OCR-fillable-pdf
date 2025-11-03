# this is the context pdf processor
import fitz as pypdf
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from graph_chunking import semantic_graph_chunking
from langchain_classic.schema import Document

class ContextPDFProcessor:
    vectordb_index = None
    model = None

    def __init__(self):
        # initializing the envs
        self._init_env()

        self.vectordb_index = self._create_vectordb_instance()
        self.model = self._create_model_instance()
        self.embeddingModel = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    @classmethod
    def _init_env(cls):
        load_dotenv()

    @classmethod
    def _create_model_instance(cls):
        model = ChatOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=128
        )

        return model

    @classmethod
    def _create_vectordb_instance(cls):
        client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "ocr-project"
        if index_name not in client.list_indexes().names():
            client.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        index = client.Index(index_name)
        return index
    
    def extract_text_from_pdf(self, filepath: str) -> str:
        doc = pypdf.open(filepath)
        pagecontextText = {}
        for i in range(len(doc)):
            pagecontextText[i+1] = doc[i].get_text()

        return pagecontextText
    
    def generate_answers(self, context, questions: dict):
        vector_store = PineconeVectorStore(index=self.vectordb_index, embedding=self.embeddingModel)
        chunks = semantic_graph_chunking(context)
        documents = [Document(page_content=text) for text in chunks]

        vector_store.add_documents(documents=documents)

        system_prompt = '''
                You are an information extraction model. Your task is to extract the exact, factual answer from the provided context — nothing more, nothing less.

                Guidelines:
                - Output only the precise value(s) that directly answer the user’s query.
                - Do NOT include explanations, reasoning, or restatements of the question.
                - If the answer is a **name**, output only the full name (first + last name).
                - If the answer is an **address**, include the complete address (street, city, state, country, postal code if present).
                - If the answer is a **number**, **date**, **ID**, or **code**, return it exactly as it appears.
                - If multiple values are present and relevant, return them as a comma-separated list.
                - If the exact answer is not found in the context, respond only with: "Answer not found in the context file".
                - Never hallucinate or infer beyond what is explicitly stated in the context.
            '''

        prompt = ChatPromptTemplate.from_messages(
            [
                ('system', system_prompt),
                ('human', "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.model, prompt)
        chain = create_retrieval_chain(vector_store.as_retriever(), question_answer_chain)

        idx = 1
        id_ans_mapping = {}
        for question, _ in questions:
            id_ans_mapping[idx] = chain.invoke({"input": question})
            idx += 1

        return id_ans_mapping
        
