import loader as loader
from langchain.document_loaders.csv_loader import CSVLoader
import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

#para csv
os.environ['OPENAI_API_KEY'] = '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'
default_doc_name = 'doc.csv'

def process_doc(
        path: str = 'C:\\Users\\Mateo\\Desktop\\tabla_Calif.csv',
        is_local: bool = False,
        question: str = 'Cuál es la nota de Zachary en Química?'
):
    loader = CSVLoader(file_path= path, csv_args={
        'delimiter': ';',
        'fieldnames': ['Nombre', 'Inglés', 'Matemática', 'Historia', 'Química', 'Física']
    })

    doc = loader.load_and_split()

    #print(doc[-1])
    db = Chroma.from_documents(doc, embedding=OpenAIEmbeddings())

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=db.as_retriever())
    #st.write(qa.run(question))
    print( "\n*** LA PREGUNTA ES:"+question, "\n****"+qa.run(question))


if __name__ == '__main__':
   process_doc()
