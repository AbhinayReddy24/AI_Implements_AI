import openai
import os
import sys

from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import runnable
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from dotenv import load_dotenv
import time

env = load_dotenv()
key = os.environ["OPENAI_API_KEY"]

start_time = time.time()

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

loader = PyPDFLoader(
    "../documents/Real-Time_Detection_of_DNS_Exfiltration_and_Tunneling_from_Enterprise_Networks-1.pdf"
)
pages = loader.load()


# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(pages)

# Embed and store splits
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=key)
vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vector_store.as_retriever()

# create retrievalQA object
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

# Creates conversational ability and document retrieval capability
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=key),
    retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
)

end_time = time.time()
print(f"Setup (loader,vector,retriever chain prepared) done in {end_time - start_time:.2f}s\n")


chat_history = []

questions = [
    "What port/protocol is used for DNS?",
    "Is that usually allowed unfettered access through enterprise firewalls?",
    "What were the final parameters used in the isolationForest model",
    "What were the final parameters used in the isolationForest model",
    "What were the final parameters used in the isolationForest model",
    "Describe what was there in the table 3",
    "Summarize the paper in a paragraph",
    "Describe figure 2 along with statistics",
    "Describe figure 2 along with statistics in 2 sentences or less",
    "Explain about the average time complexity of their scheme based on the table 5 in simple terms",
    "What were the attributes derived from the DNS and what were the attributes from FQDN, based on the output of this query form a list of features",
    "Can you describe the steps to implement the model"
]

# while True:
#     if not query:
#         query = input("Prompt: ")
#     if query in ["quit", "q", "exit"]:
#         sys.exit()

# for each question in the questions list
for question in questions:
    print(f"Question: {question}", )

    start_time = time.time()
    result = chain({"question": question, "chat_history": chat_history})
    answer = result["answer"]
    end_time = time.time()

    print(f"Ans: {answer}")
    print(f"Elapsed: {end_time - start_time:.2f}s\n")

    chat_history.append((question, answer))

