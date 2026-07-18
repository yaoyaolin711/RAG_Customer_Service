import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
for _P in [os.path.join(_REPO_ROOT, "crm_agent", "crm_agent"), os.path.join(_REPO_ROOT, "RAG_mode", "mode")]:
    if os.path.isfile(os.path.join(_P, "settings.py")):
        sys.path.append(_P)
        break

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from settings import TOP_K, LLM_MODEL_NAME, LLM_MODEL_BASE_URL, get_aliyun_api_key
from vectorstore import get_vector_store, check_milvus_connection

check_milvus_connection()
vector_store = get_vector_store()


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    docs = vector_store.similarity_search(query, k=TOP_K)
    serialized = "\n".join(f"{d.metadata.get('source')}: {d.page_content[:100]}" for d in docs)
    return serialized, docs


model = init_chat_model(
    model=LLM_MODEL_NAME, model_provider="openai", api_key=get_aliyun_api_key(), base_url=LLM_MODEL_BASE_URL
)
agent = create_agent(model, tools=[retrieve_context], system_prompt="浣犳槸閿﹀皬涓烇紝绠€娲佸洖绛斻€?)

question = "婊″灏戝厤杩愯垂锛?
print(f"闂: {question}")
resp = agent.invoke({"messages": [{"role": "user", "content": question}]})
print(f"鍥炵瓟: {resp['messages'][-1].content}")

