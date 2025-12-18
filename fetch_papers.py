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

    def search_papers_with_split(
        self,
        query: str,
        year: str,
        fields: List[str],
        base_filename: str,
        max_size_mb: int = 100
    ) -> tuple[int, int, List[int], List[str]]:
        """
        論文を検索してJSONL形式で逐次保存（ファイルサイズ制限付き）

        Args:
            query: 検索クエリ
            year: 年フィルタ（例: "2025"）
            fields: 取得するフィールドのリスト
            base_filename: 出力ファイル名のベース（拡張子なし）
            max_size_mb: 1ファイルの最大サイズ（MB）

        Returns:
            (総検索結果数, 取得件数, 引用数リスト, 出力ファイルリスト)
        """
        params = {
            "query": query,
            "year": year,
            "fields": ",".join(fields)
        }

        total_results = 0
        fetched_count = 0
        citations = []
        output_files = []
        token = None
        page = 1

        # 現在のファイル情報
        current_file_index = 1
        current_file = None
        current_file_size = 0
        max_size_bytes = max_size_mb * 1024 * 1024

        print(f"検索中: '{query}' (年: {year})")
        print(f"取得フィールド: {', '.join(fields)}")
        print(f"ファイル分割サイズ: {max_size_mb} MB")
        print("-" * 80)

        try:
            # 最初のファイルを開く
            if current_file_index == 1:
                filename = f"{base_filename}.jsonl"
            else:
                filename = f"{base_filename}_part{current_file_index}.jsonl"

            current_file = open(filename, "w", encoding="utf-8")
            output_files.append(filename)
            print(f"ファイル作成: {filename}")

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
                    # JSON行を作成
                    json_line = json.dumps(paper, ensure_ascii=False) + "\n"
                    line_size = len(json_line.encode('utf-8'))

                    # ファイルサイズをチェック
                    if current_file_size + line_size > max_size_bytes and fetched_count > 0:
                        # 現在のファイルを閉じる
                        current_file.close()
                        file_size_mb = current_file_size / (1024 * 1024)
                        print(f"\nファイル完成: {output_files[-1]} ({file_size_mb:.1f} MB)")

                        # 新しいファイルを開く
                        current_file_index += 1
                        filename = f"{base_filename}_part{current_file_index}.jsonl"
                        current_file = open(filename, "w", encoding="utf-8")
                        output_files.append(filename)
                        current_file_size = 0
                        print(f"ファイル作成: {filename}")

                    # 書き込み
                    current_file.write(json_line)
                    current_file_size += line_size
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

        finally:
            # 最後のファイルを閉じる
            if current_file:
                current_file.close()
                file_size_mb = current_file_size / (1024 * 1024)
                print(f"ファイル完成: {output_files[-1]} ({file_size_mb:.1f} MB)")

        return total_results, fetched_count, citations, output_files


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
    query = '"large language model"'
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

    # 出力ファイル名のベース
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"semantic_scholar_llm_2025_{timestamp}"

    # フェッチャーを初期化
    fetcher = SemanticScholarFetcher(api_key)

    try:
        # 論文を検索・分割保存（100MB制限）
        total, fetched_count, citations, output_files = fetcher.search_papers_with_split(
            query, year, fields, base_filename, max_size_mb=100
        )

        # 統計情報を表示
        if fetched_count > 0:
            print(f"\n{fetched_count:,} 件の論文を保存しました。")
            if len(output_files) > 1:
                print(f"ファイルは {len(output_files)} 個に分割されました:")
                for i, file in enumerate(output_files, 1):
                    file_size = os.path.getsize(file) / (1024 * 1024)
                    print(f"  {i}. {file} ({file_size:.1f} MB)")
            else:
                print(f"ファイル: {output_files[0]}")

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

            # manifest.jsonを更新
            update_manifest("llm", output_files[0], timestamp)
            print(f"\nmanifest.json を更新しました。")
        else:
            print("\n論文が見つかりませんでした。")

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        raise


def update_manifest(dataset_type: str, base_filename: str, timestamp: str):
    """
    manifest.jsonを更新して最新のファイル情報を記録

    Args:
        dataset_type: データセットタイプ（'llm' or 'vlm'）
        base_filename: ベースファイル名（最初のファイル名）
        timestamp: タイムスタンプ
    """
    manifest_file = "manifest.json"

    # 既存のmanifestを読み込む
    if os.path.exists(manifest_file):
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {}

    # 最新情報を更新
    manifest[dataset_type] = {
        "filename": base_filename,
        "timestamp": timestamp,
        "updated_at": datetime.now().isoformat()
    }

    # manifestを保存
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
