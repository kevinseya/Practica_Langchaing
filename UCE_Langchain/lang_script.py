import os

from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

import streamlit as st


os.environ['OPENAI_API_KEY'] = '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'
default_doc_name = 'doc.pdf'
default_doc_name1 = 'doc1.pdf'


def process_doc(
        path: str = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf',
        path2: str = 'https://proceedings.neurips.cc/paper_files/paper/1988/file/149e9677a5989fd342ae44213df68868-Paper.pdf',
        is_local: bool = False,
        question: str = 'Quiénes son los autores del pdf?'
):
    #_, loader = os.system(f'curl -o {default_doc_name} {path}'), PyPDFLoader(f"./{default_doc_name}") if not is_local \
    #    else PyPDFLoader(path)

    _, loader = os.system(f'curl -o {default_doc_name1} {path2}'), PyPDFLoader(f"./{default_doc_name1}") if not is_local \
        else PyPDFLoader(path2)


    doc = loader.load_and_split()
    doc2 = loader.load_and_split()

    print(doc[-1])

    #db = Chroma.from_documents(doc, embedding=OpenAIEmbeddings())
    db = Chroma.from_documents(doc2, embedding=OpenAIEmbeddings())

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=db.as_retriever())
    st.write(qa.run(question))
    print( "\n*** LA PREGUNTA ES:"+question, "\n****"+qa.run(question))


def client():
    st.title('Manage LLM with LangChain')
    uploader = st.file_uploader('Upload PDF', type='pdf')

    if uploader:
        with open(f'./{default_doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('PDF saved!!')

    question = st.text_input('Generar un resumen de 20 palabras sobre el pdf',
                             placeholder='Give response about your PDF', disabled=not uploader)

    if st.button('Send Question'):
        if uploader:
            process_doc(
                path=default_doc_name,
                is_local=True,
                question=question
            )
        else:
            st.info('Loading default PDF')
            process_doc()


if __name__ == '__main__':
   client()
   #process_doc()
