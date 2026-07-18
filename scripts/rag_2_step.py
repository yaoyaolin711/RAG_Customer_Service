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
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

from settings import TOP_K, LLM_MODEL_NAME, LLM_MODEL_BASE_URL, get_aliyun_api_key
from vectorstore import get_vector_store, check_milvus_connection

check_milvus_connection()
vector_store = get_vector_store()


@dynamic_prompt
def dynamic_prompt_fn(request: ModelRequest) -> str:
    query = request.messages[-1].content
    retrieved_docs = vector_store.similarity_search(query, k=TOP_K)

    prompt = """浣犳槸閿︿笧鍟嗗煄鐨勬櫤鑳藉鏈嶅姪鎵嬶紝鍚嶅瓧鍙敠灏忎笧銆傝鏍规嵁浠ヤ笅浠庣煡璇嗗簱妫€绱㈠埌鐨勫唴瀹瑰洖绛旂敤鎴烽棶棰樸€?
            鍥炵瓟鏃惰姘斾翰鍒囥€佺畝娲佷笓涓氥€傚鏋滀俊鎭笉瓒充互鍥炵瓟锛岃璇?鎶辨瓑锛屾垜鏆傛椂鏃犳硶鏍规嵁鐜版湁淇℃伅鍥炵瓟璇ラ棶棰樸€?
            妫€绱㈠埌鐨勫唴瀹癸細
            """
    for doc in retrieved_docs:
        prompt += f"{doc.metadata.get('source', '鏈煡')}锛歿doc.page_content}\n"

    return prompt


model = init_chat_model(
    model=LLM_MODEL_NAME,
    model_provider="openai",
    base_url=LLM_MODEL_BASE_URL,
    api_key=get_aliyun_api_key(),
)
agent = create_agent(model, middleware=[dynamic_prompt_fn])

response = agent.invoke({"messages": [
    {"role": "user", "content": "閲戝崱浼氬憳鏈変粈涔堟潈鐩婏紵"}
]})

for msg in response["messages"]:
    if isinstance(msg, AIMessage) and msg.content.strip() != "":
        print(msg.content)

