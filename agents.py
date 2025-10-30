import json
import glob
from langchain_community.document_loaders import JSONLoader
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent

FHIR_BUNDLE_JQ_SCHEMA = ".entry[]"
FHIR_BUNDLE_CONTENT_KEY = ".resource"

synthea_bundles = glob.glob("./fhir/*.json")

LLM = ChatOllama(model="llama3.2:1b") # using smaller model due to resource limitations on my old laptop 🙃

@tool
def get_fhir_data() -> list[str]:
    """Retrieves FHIR data for a set of Patients, including details about Procedures

    Returns:
        List of FHIR Resources as JSON
    """
    loaders = [
        JSONLoader(
            file_path=xpath,
            jq_schema=FHIR_BUNDLE_JQ_SCHEMA,
            content_key=FHIR_BUNDLE_CONTENT_KEY,
            is_content_key_jq_parsable=True,
            text_content=False
        )
        for xpath in synthea_bundles[:3] # limiting to just the first 3 for now
    ]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    
    return docs

fhir_retrieval_agent = create_agent(
    model=LLM,
    tools=[get_fhir_data],
    system_prompt="You retrieve patient records and summarize based on the user query"
)

response = fhir_retrieval_agent.invoke(
    {
        "messages": [
            { "role": "user", "content": "what are the top procedures performed?" }
        ]
    }
)

contents = []

for message in response['messages']:
    contents.append(message.__getattribute__("content"))

with open("agents_output.txt", "w") as fp:
    fp.write("\n".join(contents))