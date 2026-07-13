from typing import List
import requests

class Retriever:

    def __init__(self, config=None):
        """创建 TriviaQA 检索客户端。

        参数来自 dataset.retrieval；保留本地服务默认值，同时允许服务器通过
        YAML/CLI 覆盖 URL、top-k 和超时。
        """

        config = config or {}
        self.config = {
            "search_url": config.get("search_url", "http://127.0.0.1:8001/retrieve"),
            "topk": int(config.get("topk", 3)),
            "timeout": float(config.get("timeout", 30)),
        }

    def batch_search(self, queries: List[str] = None) -> List[str]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']

        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        if not queries:
            raise ValueError("Retriever.batch_search requires at least one query")
        payload = {
            "queries": queries,
            "topk": self.config["topk"],
            "return_scores": True
        }

        response = requests.post(
            self.config["search_url"],
            json=payload,
            timeout=self.config["timeout"],
        )
        response.raise_for_status()
        body = response.json()
        if "result" not in body:
            raise ValueError("Retriever response is missing the 'result' field")
        return body

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):

            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
