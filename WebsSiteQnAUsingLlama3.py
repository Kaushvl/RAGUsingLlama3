import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from ollama import chat
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def GetWebText(strUrl:str):
    '''
    Purpose : To extract strText from a website 
    input : (1) url = Url of the site to extract strText
    Output : (1) strWebcontent : content of website in string format
    '''
    objLoader = RecursiveUrlLoader(
        url=strUrl, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
    )
    objData = objLoader.load()

    strWebcontent = objData[0].page_content

    with open('strData.txt', 'w') as file:
        file.write(strWebcontent)

    return  strWebcontent



def GetTextChunks(strText:str):
    '''
        Purpose : To create chunks from a long string
        input : (1) strText = string for which chunks has to be created
        Output : (1) lsChunks : chunks of input string
    '''
    objTextSplitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    lsChunks = objTextSplitter.split_text(strText)

    return lsChunks


def GetVectorStore(lsTextChunks:list):
    '''
        Purpose : To create vector store using list of chunks of original text content
        input : (1) lsTextChunks : list of chunks of original text content to be saved in vector store
        output : (1) None
    '''
    objEmbeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(lsTextChunks, embedding=objEmbeddings)
    vector_store.save_local("faiss_index")


def UserInput(strUserquestion:str):
    '''
        Purpose : To give answer from relevent context 
        input : (1) strUserquestion : Use question to get answer 
        output : (1) None
    '''
    objEmbeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", objEmbeddings)
    strContext = new_db.similarity_search(strUserquestion)

    # Separate user question and context
    context = strContext
    question = strUserquestion

    prompt_template = f"""
    Answer the question only from the provided context in simple words, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question:\n {question}
    """
    
    objCompletion = chat(model='llama3:8b-instruct-q8_0',messages=[{"role":"assistant","content":prompt_template}])

    st.write("Reply: ", objCompletion['message']['content'])




def main():
    st.set_page_config("Chat Website")
    st.header("Chat With url usnig Local LLAMA3")

    strUserquestion = st.text_input("Ask a Question from the PDF Files")

    if strUserquestion:
        UserInput(strUserquestion)

    with st.sidebar:
        st.title("Menu:")
        strWebUrl = st.text_input("Enter the website url")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = GetWebText(strWebUrl)
                lsTextChunks = GetTextChunks(raw_text)
                GetVectorStore(lsTextChunks)
                st.success("Done")



if __name__ == "__main__":
    main()