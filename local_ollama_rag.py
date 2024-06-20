# %% [markdown]
# ## Ingesting PDF

# # %%
# %pip install --q unstructured langchain
# %pip install --q "unstructured[all-docs]"

# %%
# !pip install langchain-community

# %%
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

# %%
local_path = "WEF_The_Global_Cooperation_Barometer_2024.pdf"

# Local PDF file uploads
if local_path:
  loader = UnstructuredPDFLoader(file_path=local_path)
  data = loader.load()
else:
  print("Upload a PDF file")

# %%
# Preview first page
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
local_model = "phi3"
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


