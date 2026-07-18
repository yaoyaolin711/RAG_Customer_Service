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
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import tool

from settings import TOP_K, LLM_MODEL_NAME, LLM_MODEL_BASE_URL, get_aliyun_api_key
from vectorstore import get_vector_store, check_milvus_connection

check_milvus_connection()
vector_store = get_vector_store()


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """鏍规嵁鐢ㄦ埛鐨勭數鍟嗙浉鍏抽棶棰樻绱㈤敠涓炲晢鍩庣煡璇嗗簱鍗忓姪鍥炵瓟"""
    retrieved_docs = vector_store.similarity_search(query, k=TOP_K)
    serialized = ""
    for doc in retrieved_docs:
        serialized += f"鏂囦欢鍚嶏細{doc.metadata.get('source', '鏈煡')} \n 鍐呭锛歿doc.page_content}\n\n"
    return serialized, retrieved_docs


model = init_chat_model(
    model=LLM_MODEL_NAME,
    model_provider="openai",
    api_key=get_aliyun_api_key(),
    base_url=LLM_MODEL_BASE_URL,
)
prompt = """浣犳槸閿︿笧鍟嗗煄鐨勬櫤鑳藉鏈嶅姪鎵嬶紝鍚嶅瓧鍙敠灏忎笧銆傝鍏堣皟鐢ㄥ伐鍏蜂粠鐭ヨ瘑搴撴绱笌鐢ㄦ埛闂鐩稿叧鐨勫唴瀹癸紝鍐嶆牴鎹绱㈢粨鏋滃洖绛斻€?
            鍥炵瓟鏃惰姘斾翰鍒囥€佺畝娲佷笓涓氥€傚鏋滀俊鎭笉瓒充互鍥炵瓟锛岃璇?鎶辨瓑锛屾垜鏆傛椂鏃犳硶鏍规嵁鐜版湁淇℃伅鍥炵瓟璇ラ棶棰樸€?
            """
agent = create_agent(model, tools=[retrieve_context], system_prompt=prompt)

for stream_mode, response in agent.stream(
        {"messages": [{"role": "user", "content": "閿︿笧鍟嗗煄7澶╂棤鐞嗙敱閫€璐ф€庝箞鐢宠锛?}]},
        stream_mode=["messages", "values"]):
    if stream_mode == "messages":
        if isinstance(response[0], AIMessageChunk):
            print(response[0].content, end="", flush=True)
    elif stream_mode == "values":
        if isinstance(response["messages"][-1], AIMessage):
            for msg in response["messages"]:
                if isinstance(msg, ToolMessage):
                    for doc in msg.artifact:
                        print("---" * 10)
                        print(doc.metadata.get("source", "鏈煡"))
                        print(doc.page_content)
                        print("---" * 10)

