import os
import sys

import pytest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
for _P in [os.path.join(_REPO_ROOT, "crm_agent", "crm_agent"), os.path.join(_REPO_ROOT, "RAG_mode", "mode")]:
    if os.path.isfile(os.path.join(_P, "settings.py")):
        sys.path.append(_P)
        break

from settings import TOP_K
from vectorstore import get_vector_store, check_milvus_connection


class TestVectorStore:
    @pytest.fixture(autouse=True)
    def setup(self):
        check_milvus_connection()
        self.vector_store = get_vector_store()

    def test_similarity_search(self):
        query = "閿︿笧鍟嗗煄婊″灏戝厤杩愯垂锛?
        retrieved_docs = self.vector_store.similarity_search(query, k=TOP_K)

        assert len(retrieved_docs) > 0

        for doc in retrieved_docs:
            print("---" * 10)
            print(doc.metadata.get("source", "鏈煡"))
            print(doc.page_content)
            print("---" * 10)

