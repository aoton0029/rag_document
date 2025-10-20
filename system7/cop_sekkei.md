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


まず意図の確認をお願いします。自動評価だけでよいですか、それとも人手評価（ラベル付け／レビュー）も混ぜますか？また以下を教えてください：
- 使用中の llama_index のバージョン（分かれば）
- 使う LLM（OpenAI / local / Llama など）
- データの種類と規模（FAQ/短文、長文、複数ドキュメント、PDF 等）
- 「精度向上」の主目的（正確さ／根拠の有無／返答の一貫性／速度など）
- 既に「正解回答」や「関連文書ラベル（relevance judgments）」があるか

上記が分かれば、より具体的な評価スクリプトやチューニング手順を出します。ここではラフな評価フレームワークと実践的手順を示します。

1) 全体方針（評価の流れ）
- データ用意：検証用クエリセット + 正解回答（ゴールド） + 各クエリに対する関連チャンクのラベル（あれば理想）。
- 分離：Train（チューニング） / Validation（ハイパー）/ Test（最終評価）に分ける。
- 層別評価：retriever 単体評価、reranker（あるなら）評価、synthesizer（応答合成）単体評価、エンドツーエンド（QueryEngine）評価。
- 自動評価 + 人手評価の組合せ：自動でスコアを回して候補絞り、人手で最終的に品質を判断。

2) Retriever の評価（目的：関連チャンクが上位に来るか）
- 指標：Recall@k（必須）、Precision@k、MRR（Mean Reciprocal Rank）、MAP、NDCG。
  - recall@k = (クエリに対して少なくとも1つゴールドチャンクが top-k に入っている割合)
  - MRR = 1 / rank_of_first_relevant averaged
- 実施方法：
  - まず大量の候補（fetch_k）を取っておき、ラベルと照合して指標を算出。
  - 異なる埋め込みモデル／ベクトルストア／k を比較。
- ツール：pytrec_eval、ir-measures
- チューニング対象：embedding model, index type, chunk_size, overlap, k, MMR λ, metadata filter

3) Reranker（ある場合）の評価
- 指標：同じく MRR, MAP, Precision@k、または pairwise accuracy（正のペアが上になる割合）
- 実施方法：retriever の上位候補を reranker にかけ、スコア順で再評価して指標計算。
- トレーニング：pairwise（RankNet/hinge）や listwise 損失を用いると効果的。

4) Response synthesizer（合成／生成）評価
- 指標（自動）：
  - QA タイプ：Exact Match (EM), F1（SQuAD スタイル）
  - 要約タイプ：ROUGE (R1/R2/L)、BLEU（必要なら）、BERTScore、BLEURT（品質の深い評価）
  - 根拠評価：citation coverage（回答内に提示した出典がゴールドに含まれる割合）、overlap-based recall（回答の根拠となるトークンが取得チャンクに含まれているか）
  - 信頼性（正確性）チェック：NLI/entailment ベースで答えが取得チャンクに含まれているかを判定（contradiction/neutral/entailment）
- 人手評価：正確性、役立ち度、明瞭性、根拠の妥当性（5点リッカートなど）、および hallucination の有無判定
- 実施方法：
  - 自動：各クエリで生成回答とゴールドを比較して EM/F1/ROUGE 等を計算。
  - Faithfulness：生成文の主張を個々の参照文と NLI で検証。あるいは LLM に「この回答のどの文が取得したソースに基づくか」を抽出させる。
  - 記録：生成毎に使用されたチャンクの ID とスコアをログに残す（後で照合・解析）。
- チューニング対象：synth strategy（stuff/map_reduce/refine/tree）、prompt 設計、max_tokens, temperature、top_p、chain-of-thought の有無、引用フォーマット

5) エンドツーエンド評価（QueryEngine）
- 指標：タスクに応じて EM/F1（QA）、Accuracy（分類）、Mean Opinion Score（MOS）や人手評価。
- 根拠ベース評価：回答と共に返した citations が正解文献に一致する割合／回答が参照したチャンクに含まれる情報だけで正しいか。
- レイテンシとコストの評価：平均応答時間、API コール回数、トークンコスト

6) ハルシネーション／信頼性対策の評価
- 自動手法：
  - NLIベース検証：生成文とソースのエンタイトルメント判定（transformers の NLI モデル使用）
  - QA逆検証：生成した答えから新たな質問を生成し、取得チャンクで答えられるかチェック
  - Lexical overlap / evidence coverage：回答中の事実表現が取得チャンクにどれだけマッチするか
- 人手：サンプルをランダム抽出して false claim をラベリング

7) 実験設計（A/B, ablation, sweep）
- 比較軸：retriever type (dense / sparse / hybrid)、k、synth strategy（stuff/map_reduce/refine）、reranker ON/OFF、chunk_size
- 手順：
  - 固定 seed で各構成を複数回評価（確率的要素がある場合）
  - 結果を表にまとめ（自動指標 + 人手スコア + レイテンシ + コスト）
  - 統計検定（t-test / bootstrap）で差の有意性確認
- ハイパーパラメータ探索：Optuna や grid search（k, fetch_k, MMR λ, temperature など）

8) モニタリング／運用での評価
- ログ：クエリ、retrieved ids, scores, final answer, LLM temperature, tokens used
- 指標の継続監視：weekly recall@k, hallucination_rate, avg_latency
- ドリフト検出：クエリ分布や retrieval performance の下落をウォッチ
- A/B テスト：新戦略を一部トラフィックで運用し、ユーザーメトリクス（満足度、解決率）を比較

9) 実践で使えるツールとライブラリ
- retrieval eval: pytrec_eval, ir-measures
- text eval: rouge_score, sacrebleu, bert-score, BLEURT
- NLI/entailment: transformers (roberta-large-mnli 等)
- ベクトル検索: FAISS, Milvus, Weaviate, chroma
- ライブラリ: sentence-transformers, datasets (HuggingFace), pandas, scikit-learn
- ハイパー探索: Optuna

10) 簡単な実装例（考え方のスニペット）
- Retriever の recall@k（擬コード）
  - ① 各クエリで top_k を取得 → 候補 id リスト
  - ② ゴールド関連チャンク id と照合して recall@k を計算
- QA の EM / F1（Python 概要）
  - 正規化（lower, strip, punctuation除去）
  - EM = normalized_pred == normalized_gold
  - F1 = token-level F1（precision/recall/harmonic）

（必要なら実コードを用意します。pytrec_eval の使い方や、NLI を使った自動faithfulnessチェッカのサンプルも出せます）

11) 優先度の提案（短期→中期→長期）
- 短期：retriever の recall@k を改善（適切な chunk_size, embedding model, fetch_k）。自動指標で候補を絞る。
- 中期：reranker を導入して上位の品質を担保。synth strategy を map_reduce / refine に切替えて出力品質を上げる。
- 長期：人手評価のラベルを増やし、end-to-end 学習（retrieval + rerank の学習やRL-based fine-tune）や出典付き応答の堅牢化を行う。

最後に：具体的なコード（retrieval 評価スクリプト、LLM を使った faithfulness チェッカ、map_reduce の評価例）を出しますか？それともまずは「どの指標を採用するか」のテンプレート（CSV でログを取る項目等）を作りましょうか。どれを優先しますか？