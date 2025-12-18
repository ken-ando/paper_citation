"""
Semantic Scholar API を使用して2025年に発表された
"Large language model"を含む論文を取得し、JSONL形式で保存する。
"""

import os
import time
import json
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime


class SemanticScholarFetcher:
    """Semantic Scholar APIから論文データを取得するクラス"""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

    def __init__(self, api_key: Optional[str] = None):
        """
        初期化

        Args:
            api_key: Semantic Scholar APIキー（環境変数 SEMANTIC_SCHOLAR_API_KEY から取得可能）
        """
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})

        # レート制限: APIキーありの場合は1 RPS、なしの場合はより慎重に
        self.rate_limit_delay = 1.1  # 秒
        self.last_request_time = 0

    def _wait_for_rate_limit(self):
        """レート制限を守るための待機"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(
        self,
        params: Dict[str, Any],
        max_retries: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        APIリクエストを実行（指数バックオフ付き）

        Args:
            params: クエリパラメータ
            max_retries: 最大リトライ回数

        Returns:
            APIレスポンス（JSON）
        """
        for attempt in range(max_retries):
            # リトライ時も含めて毎回レート制限を守る（全エンドポイント合算1RPS）
            self._wait_for_rate_limit()

            try:
                response = self.session.get(self.BASE_URL, params=params, timeout=30)

                # レート制限エラーの場合は指数バックオフ
                if response.status_code == 429:
                    wait_time = (2 ** attempt) * self.rate_limit_delay
                    print(f"レート制限に達しました。{wait_time:.1f}秒待機します...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                print(f"リクエストエラー (試行 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * self.rate_limit_delay
                    time.sleep(wait_time)
                else:
                    raise

        return None

    def search_papers_streaming(
        self,
        query: str,
        year: str,
        fields: List[str],
        output_file
    ) -> tuple[int, int, List[int]]:
        """
        論文を検索してJSONL形式で逐次保存

        Args:
            query: 検索クエリ
            year: 年フィルタ（例: "2025"）
            fields: 取得するフィールドのリスト
            output_file: 開かれたファイルハンドル

        Returns:
            (総検索結果数, 取得件数, 引用数リスト)
        """
        params = {
            "query": query,
            "year": year,
            "fields": ",".join(fields)
        }

        total_results = 0
        fetched_count = 0
        citations = []
        token = None
        page = 1

        print(f"検索中: '{query}' (年: {year})")
        print(f"取得フィールド: {', '.join(fields)}")
        print("-" * 80)

        while True:
            # トークンがある場合はパラメータに追加
            if token:
                params["token"] = token

            print(f"ページ {page} を取得中...", end=" ", flush=True)

            result = self._make_request(params)
            if not result:
                break

            # 初回のみ総件数を取得
            if page == 1:
                total_results = result.get("total", 0)
                print(f"\n総検索結果数: {total_results:,} 件")
                print("-" * 80)

            # データを逐次書き込み
            papers = result.get("data", [])
            for paper in papers:
                json.dump(paper, output_file, ensure_ascii=False)
                output_file.write("\n")
                fetched_count += 1

                # 引用数の統計用
                if paper.get("citationCount") is not None:
                    citations.append(paper["citationCount"])

            print(f"{len(papers)} 件取得（累計: {fetched_count:,} 件）")

            # 次のページのトークンを取得
            token = result.get("token")
            if not token:
                print("\nすべてのページを取得しました。")
                break

            page += 1

        return total_results, fetched_count, citations



def main():
    """メイン処理"""
    # APIキーの確認
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        print("警告: SEMANTIC_SCHOLAR_API_KEY 環境変数が設定されていません。")
        print("レート制限が厳しくなる可能性があります。")
        response = input("続行しますか？ (y/N): ")
        if response.lower() != "y":
            print("中止しました。")
            return

    # 検索パラメータ
    query = '("large language model" | "large language models")'
    year = "2025"
    fields = [
        "paperId",
        "title",
        "abstract",
        "year",
        "citationCount",
        "publicationDate",
        "authors",
        "url",
        "venue",
        "publicationTypes"
    ]

    # 出力ファイル名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"semantic_scholar_llm_2025_{timestamp}.jsonl"

    # フェッチャーを初期化
    fetcher = SemanticScholarFetcher(api_key)

    try:
        # ファイルを開いて論文を検索・逐次書き込み
        with open(output_file, "w", encoding="utf-8") as f:
            total, fetched_count, citations = fetcher.search_papers_streaming(
                query, year, fields, f
            )

        # 統計情報を表示
        if fetched_count > 0:
            print(f"\n{fetched_count:,} 件の論文を {output_file} に保存しました。")

            print("\n" + "=" * 80)
            print("統計情報")
            print("=" * 80)
            print(f"総検索結果数: {total:,} 件")
            print(f"取得件数: {fetched_count:,} 件")

            # 引用数の統計
            if citations:
                print(f"\n引用数統計:")
                print(f"  - 最大: {max(citations):,} 回")
                print(f"  - 最小: {min(citations):,} 回")
                print(f"  - 平均: {sum(citations) / len(citations):.1f} 回")
        else:
            print("\n論文が見つかりませんでした。")

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
