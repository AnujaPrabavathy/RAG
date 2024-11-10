import warnings
warnings.filterwarnings('ignore')
from filetype import guess
from langchain_unstructured import UnstructuredLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'




def detect_document_type(document_path):
    
    guess_file = guess(document_path)
    file_type = ""
    image_types = ['jpg', 'jpeg', 'png', 'gif']
    
    if(guess_file.extension.lower() == "pdf"):
        file_type = "pdf"
        
    elif(guess_file.extension.lower() in image_types):
        file_type = "image"
        
    else:
        file_type = "unkown"
        
    return file_type




document_path = "folder path"
#article_information_path = "./data/zoumana_article_information.png"

print(f"Document Type: {detect_document_type(document_path)}")
#print(Document Type: {detect_document_type(article_information_path)}")


def extract_text_from_pdf(pdf_file):
    
    loader = UnstructuredFileLoader(pdf_file)
    documents = loader.load()
    pdf_pages_content = '\n'.join(doc.page_content for doc in documents)
    
    return pdf_pages_content

def extract_text_from_image(image_file):

    loader = UnstructuredImageLoader(image_file)
    documents = loader.load()
    
    image_content = '\n'.join(doc.page_content for doc in documents)
    
    return image_content

def extract_file_content(file_path):
    
    file_type = detect_document_type(file_path)
    
    if(file_type == "pdf"):
        loader = UnstructuredFileLoader(file_path)
        
    elif(file_type == "image"):
        loader = UnstructuredImageLoader(file_path)
        
    documents = loader.load()
    documents_content = '\n'.join(doc.page_content for doc in documents)
    
    return documents_content

doc_content = extract_file_content(document_path)


nb_characters = 400

print(f"First {nb_characters} Characters of the Paper: \n{doc_content[:nb_characters]}...")
print("---"*5)



from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(        
    separator = "\n\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)


research_paper_chunks = text_splitter.split_text(doc_content)

print(f"# Chunks in Research Paper: {len(research_paper_chunks)}")


os.environ["OPENAI_API_KEY"] = "OPEN API KEY"

embeddings = OpenAIEmbeddings()


def get_doc_search(text_splitter):
    
    return FAISS.from_texts(text_splitter, embeddings)

doc_search_paper = get_doc_search(research_paper_chunks)
print(doc_search_paper)


from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(OpenAI(), chain_type = "map_rerank",  
                      return_intermediate_steps=True)

def chat_with_file(file_path, query):
    
    file_content = extract_file_content(file_path)
    file_splitter = text_splitter.split_text(file_content)
    
    document_search = get_doc_search(file_splitter)
    documents = document_search.similarity_search(query)
    
    results = chain({
                        "input_documents":documents, 
                        "question": query
                    }, 
                    return_only_outputs=True)
    
    print(results)
    results = results['intermediate_steps'][0]
    
    return results


query = "Your question"

results = chat_with_file(document_path, query)

answer = results["answer"]
confidence_score = results["score"]

print(f"Answer: {answer}\n\nConfidence Score: {confidence_score}")