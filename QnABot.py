from langchain_community.llms import CTransformers #dung de load mo hinh LLM
from langchain.chains import RetrievalQA #tao chuoi truy xuat thong tin tu vector db de sinh ra cau tra loi
from langchain.prompts import PromptTemplate #dinh nghia template cho prompt input va output
from langchain_community.embeddings import GPT4AllEmbeddings 
from langchain_community.vectorstores import FAISS #truy van vector db


#Cau hinh
model_file = "models/vinallama-7b-chat.Q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

#Load LLM
def load_llm(model_file): #tai model LLM tu file
    llm = CTransformers(
        model = model_file,
        model_type = "llama",
        max_new_tokens = 1024,
        temperature = 0.01
    )

    return llm


#Tao prompt template
def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables = ["context", "question"])
    return prompt

def create_qa_chain(prompt, llm, db): #tao chuoi truy xuat QA
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs = {"k": 3}, max_tokens_limit = 1024), #dung de truy van, dua ra 3 van ban gan nhat voi cau query
        return_source_documents = False, #k can quan tam la van ban nao, chi can dua ra cau trl
        chain_type_kwargs = {'prompt': prompt}
    )
    return llm_chain

#Read tu vectorDB:
def read_vector_db():
     #Embedding:
    embedding_model = GPT4AllEmbeddings(model_file='models/all-MiniLM-L6-v2-f16.gguf')
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db 


#Bat dau test:
db = read_vector_db()
llm = load_llm(model_file)

#Tao prompt:
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu không biết câu trả lời hãy nói không biết, đừng cố tạo ra câu trả lời \n
{context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant
"""

prompt = create_prompt(template)

llm_chain = create_qa_chain(prompt, llm, db)

#Chay chain:
question = "Các loại data types"
response = llm_chain.invoke({"query": question})
print(response)

                                        

