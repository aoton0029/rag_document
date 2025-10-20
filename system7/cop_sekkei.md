どの深さで説明しましょうか？（1) 概要＋選び方、(2) 具体的なコード例、(3) 両方）  
使っている llama_index のバージョンや使いたい LLM（OpenAI / local / など）、対象データの種類（短文・長文・マルチドキュメント）を教えていただければ、具体例を出します。  

ここではまず要点を短く整理します — retriever 戦略、response_synthesizer 戦略、QueryEngine 戦略ごとに目的・代表的手法・選び方のポイントを示します。

1) Retriever（検索・取得）戦略 — 目的：関連文書（チャンク）を効率的に取り出す
- ベクター類似検索（dense retrieval）
  - 埋め込み（Embedding）でベクトルストア（FAISS, Milvus, Weaviate 等）に格納し、cosine／dot で上位 k を取得。
  - 長所：意味的類似性に強い（パラフレーズや語彙差に強い）。  
  - 短所：埋め込み品質に依存。メモリ・ストア管理が必要。
- キーワード／BM25（sparse retrieval）
  - 文字列一致や単語頻度に基づく順位付け。短いファクト照合で有効。
  - 長所：確実なキーワード一致に強い。実装・解釈が容易。  
  - 短所：語彙違いに弱い。
- ハイブリッド（sparse + dense）
  - 両者を組み合わせて候補を広く取り、再ランキングすることが多い。実運用で堅牢。
- フィルタリング（metadata filter）
  - 日付、タグ、言語などで検索範囲を絞る。精度向上に重要。
- 再ランキング（reranker）
  - まず広く取得（k large）→ LLMやクロスエンコーダで再評価して top-n に絞る（精度向上）。
- MMR（Maximal Marginal Relevance）
  - 多様性を取り入れたいとき。類似性と既取得の重複を抑える。
- チャンク設計（chunk size / overlap）
  - チャンクが小さすぎると文脈が切れる、大きすぎるとノイズ・コスト増。典型は 200–800 トークン + 10–20% overlap。

選び方ポイント：
- 質問が「事実検索（ファクト）」→ sparse + metadata が手早い。
- 意味的な質問（会話、要約、類推）→ dense が有利。
- 実運用での安定性→ハイブリッド＋reranker を推奨。

2) Response synthesizer（応答合成）戦略 — 目的：取得した複数のチャンクを LLM に渡して最終回答を作る
- Stuff（全部突っ込む）
  - 取得した文書をそのままプロンプトに全部詰めて生成。
  - 長所：シンプル、情報損失が少ない（小数の短いチャンク向け）。  
  - 短所：トークン制約・文脈ノイズ。長文では不可。
- Map-Reduce（map_reduce）
  - 各チャンクで個別に要約/答え（map）→ それらを統合して最終回答（reduce）。
  - 長所：大規模データにスケール。中間の要約でノイズ低減。  
  - 短所：設計次第で情報ロス。reduce のプロンプト設計が重要。
- Refine（逐次改善）
  - 初回回答を作り、順次チャンクを読み込みながら回答を改善・追記していく。
  - 長所：段階的に情報を反映できる。重要な追加情報を取りこぼしにくい。  
  - 短所：実行コスト高、管理がやや複雑。
- Tree-summarize / hierarchical
  - 大量ドキュメントをツリー構造で要約→上位要約をさらに統合。
  - 長所：大規模集合を階層的に圧縮可能。  
  - 短所：実装複雑、チューニング必要。
- Rerank-and-answer（retrieve→rerank→LLM)
  - まず候補を取得して再ランキング（cross-encoder 等）、上位をLLMに渡す。  

選び方ポイント：
- 取得チャンクが少ない（数件）→ stuff が最速。
- 取得チャンクが多数・長い→ map_reduce / tree / refine を検討。
- 一貫性と根拠提示が重要→ map_reduce や refine で「根拠付き回答」を設計する。

3) QueryEngine 戦略 — 目的：Retriever と ResponseSynthesizer を組み合わせ、クエリから最終応答までの流れを定義
- RetrieverQueryEngine（単純フロー）
  - Retriever で候補取得 → 指定の synth strategy で応答を作る。最も基本的な構成。
- Composition / Pipeline QueryEngine
  - 複数の QueryEngine を組み合わせ（例えば、最初はベクターで探索、見つからなければ外部APIを叩く等）。分岐・ルールベース処理に強い。
- Tool-enabled QueryEngine（外部ツールやチェイン統合）
  - LLM を「コントローラ」にして DB、検索、計算など複数ツールを呼ぶパターン。
- Streaming / Async QueryEngine
  - ストリーミング応答や長時間処理の逐次返却が必要な場面で有用。
- Hybrid QueryEngine（ranking + rerank + synth）
  - 大きく候補を集めてから再ランキング→統合する実運用向けの堅牢フロー。

QueryEngine の設計ポイント：
- レイテンシ重視 → Retriever の k を小さく、stuff や単一パス合成を使う。
- 精度・根拠重視 → k を大きく取り reranker + map_reduce/refine を使う（コスト増）。
- 可観測性（どのソースから根拠を取ったか）を重視 → metadata と root-citation を出せる合成戦略を使う。

実務でよく使う具体的な組合せ（典型）
- QA（短いドキュメント群）: dense retriever (k=5) + reranker (k=3) + stuff
- QA（大規模コーパス）: dense retriever (k=50) + MMR/re-ranking → map_reduce（map:要約, reduce:統合）
- FAQ検索（高速）: BM25 + metadata filter + stuff
- 法的/医療など根拠提示必須: hybrid retriever + reranker + refine + 出典付き応答

チューニングでよく触るパラメータ
- k（top_k）：最初の候補数
- fetch_k / candidate_k：再ランキング用に多めに取る値
- chunk_size / overlap：チャンク分割方針
- reranker スコア閾値：低スコアを捨てる
- MMR の λ（類似性 vs 多様性の重み）
- LLM の temp / max_tokens：生成の多様さと長さ

もしコード例（実際の llama_index API を使ったサンプル）を見たいなら、使っているバージョンとどの LLM（OpenAI API, local LLM など）を教えてください。具体的なデータ構造（PDF、Webページ、DB）も教えていただければ、最適な retriever + synthesizer のテンプレートを提示します。どれを優先しますか？