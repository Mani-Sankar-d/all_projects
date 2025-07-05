from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_chroma import Chroma
import os
prompt = PromptTemplate(
    template="give the answer to this question {query} based on the following context and if context is unsufficient say dont know\n {context}",
    input_variables=['query','context']
)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def run(query,id):
    db_path=f'./chroma_db/{id}'
    if(os.path.exists(db_path)):
        vector_store = Chroma(persist_directory=db_path,embedding=embeddings)
    else:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(id)
        except Exception as e:
            return f"couldnt retrieve transcript {str(id)}"
        content = " ".join([chunk['text'] for chunk in transcript])
        docs = [Document(page_content=content,metadata={"src":"yt"})]
        chunks = splitter.split_documents(documents=docs)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding= embeddings,
            persist_directory = db_path
        )
        vector_store.persist() 
    retriever = vector_store.as_retriever(search_kwargs={'k':3})
    parser = StrOutputParser()
    model = ChatHuggingFace(llm=llm)
    chain = RunnableParallel({'query':RunnablePassthrough(),'context':retriever | (lambda docs : "\n\n".join(doc.page_content for doc in docs))}) | prompt | model | parser
    return chain.invoke(query)
