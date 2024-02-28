from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from dotenv import load_dotenv
#load environment variables
load_dotenv()
# creates embeddings
embeddings= OpenAIEmbeddings()
#splits the doc by 500 characters
text_splitter= CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=200
)
#reads the pdf
loader = PyPDFLoader("genAI.pdf")
docs = loader.load_and_split( text_splitter=text_splitter)
#costs some money to run this
db= Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="database"
)
results = db.similarity_search("",k=1)

# print(results)
for doc in results:
    print(doc.page_content)
    print('\n')


