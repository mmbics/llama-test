from langchain import hub
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader(
    path='/home/aung/projects/py-projects/llama-test', 
    glob="**/*.py",
    show_progress=True,
    use_multithreading=True
)

codes = loader.load()
len(codes)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_splits = text_splitter.split_documents(codes)

vectorstore = Chroma.from_documents(
    documents=all_splits, 
    embedding=OllamaEmbeddings(model='llama3', show_progress=True),
    persist_directory='./chroma_db'
)

llm = Ollama(model='llama3')

retriever = vectorstore.as_retriever()

def format_codes(codes):
    return "\n\n".join([code.page_content for code in codes])

rag_prompt = hub.pull("rlm/rag-prompt")
qa_chain = (
    {"context": retriever | format_codes, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

question = "Why write.py"
qa_chain.invoke(question)


