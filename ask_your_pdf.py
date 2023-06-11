from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI 
from PIL import Image
import easyocr


# add ocr to read the text from the image
reader = easyocr.Reader(['en'])
load_dotenv()
st.set_page_config(page_title="Ask your PDF")
st.header("Ask your PDF 💬")

file = st.file_uploader("Upload your PDF", type="pdf")

# cash the create_knowledge_base


@st.cache_data
def create_knowledge_base(file):
    print("Creating knowledge base...")
    df_reader = PdfReader(file)
    text = ""
    for page in df_reader.pages:
        text += page.extract_text()

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)

    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base


if file is not None:
    user_question = st.text_input("Ask a question about your PDF:")
    # add camera input
    image = st.camera_input("Take a picture of your question")

    if st.button("Ask") and user_question:
        with st.spinner("Answering your question..."):
            knowledge_base = create_knowledge_base(file)

            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            st.write(response)

    if image is not None:
        st.image(image, caption="Your image", use_column_width=True)
        image = Image.open(image)
    
        with st.spinner("Reading your question..."):
            text = reader.readtext(image)
            user_question = ""
            for i in text:
                user_question += i[1] + " "
            input_question = st.text_input("Your question:", user_question)
            if st.button("Ask", key="text_from_camera") and input_question:
                with st.spinner("Answering your question..."):
                    knowledge_base = create_knowledge_base(file)

                    docs = knowledge_base.similarity_search(input_question)

                    llm = OpenAI()
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=input_question)
                    st.write(response)



