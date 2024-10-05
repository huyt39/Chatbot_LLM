from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate

#Cau hinh
model_file = "models/vinallama-7b-chat.Q5_0.gguf"

#Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model = model_file,
        model_type = "llama",
        max_new_tokens = 1024,
        temperature = 0.01
    )

    return llm


#Tao prompt template
def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables = ["question"])
    return prompt



#Chay thu chain
template = """
<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
Hello world!<|im_end|>
<|im_start|>assistant"""

prompt = create_prompt(template)
llm = load_llm(model_file)

response = prompt | llm
question = "1+1=?"
result = response.invoke({"question": question})
print(result)