import getpass
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


model = init_chat_model("gemini-2.0-flash", model_provider="google_genai", temperature=1)


messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("What time is it?"),
]


# print(model.invoke(messages).content)

# Streaming

# for token in model.stream(messages):
    # print(token.content, end="|")

# PROMPT TEMPLATE FROM MESSAGES
print("=========================PROMPT TEMPLATE FROM MESSAGES=========================")


from langchain_core.prompts import ChatPromptTemplate
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "arabic", "text": "hi!"})
print(prompt.to_messages())
response = model.invoke(prompt)
print(response.content)

# PROMPT TEMPLATE FROM TEMPLATES
print("=========================PROMPT TEMPLATE FROM TEMPLATES=========================")


from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")

prompt = prompt_template.invoke({"topic": "cats"})
response = model.invoke(prompt)
print(response.content)

prompt_template = PromptTemplate.from_template("Give me one good facts about {wrestler_name} in {wrestling_company}")
prompt =  prompt_template.invoke({"wrestler_name":"roman reigns" ,"wrestling_company":"wwe"})
response = model.invoke(prompt)
print(response.content)


# RUNNABLES (simples way)
print("=========================RUNNABLES=========================")



from langchain_core.output_parsers import StrOutputParser

runnable_chain = prompt_template | model | StrOutputParser()
print(runnable_chain.invoke({"wrestler_name":"cody rhodes" ,"wrestling_company":"aew"}))

# RUNNABLES (cascading chains)


career_summary_prompt = PromptTemplate.from_template("Give me a career summary of {wrestler_name}")
composed_chain = {"wrestler_name":runnable_chain} | career_summary_prompt | model | StrOutputParser()
print(composed_chain.invoke({"wrestler_name":"Dean Ambrose","wrestling_company":"aew"})) # this inputs are for first runnable chain





from langchain_core.runnables import RunnableLambda
