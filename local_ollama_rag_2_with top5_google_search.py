# %% [markdown]
# ## collecting text 

# %%
# %pip install --q unstructured langchain
# %pip install --q "unstructured[all-docs]"

# %%
# ! pip install googlesearch-python

# %%
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

# %%
# !pip install bs4

# %%
# from googlesearch import search
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import re
def scrape_search_results(query):
    # Perform a Google search and get the top 5 results
    search_results = search(query)

    # Collect text from the top 5 search results
    top_results_text = []
    for url in search_results:
        # Fetch the content of each search result URL
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find and collect text from relevant HTML elements (e.g., paragraphs)
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        
        # Add the result text to the list
        top_results_text.append(text)
    
    return top_results_text

# Example usage:
search_query = "Local News"
results = scrape_search_results(search_query)
# for idx, result in enumerate(results, start=1):
#     print(f"Result {idx}: {result}"))

# %%
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# %%
data = [Document(result) for result in results]

# %%
data[0].page_content

# %% [markdown]
# ## Vector Embeddings

# %%
# !ollama pull nomic-embed-text

# %%
# !ollama list

# %%
# %pip install --q chromadb
# %pip install --q langchain-text-splitters

# %%
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# %%
# Split and chunk 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# %%
# Add to vector database
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
    collection_name="local-rag"
)

# %% [markdown]
# ## Retrieval

# %%
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# %%
# LLM from Ollama
local_model = "llama3"
llm = ChatOllama(model=local_model)

# %%
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# %%
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# %%
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# %%
chain.invoke(input(""))

# %%
chain.invoke("What are the 5 pillars of global cooperation?")

# %%
# Delete all collections in the db
vector_db.delete_collection()

# %%



