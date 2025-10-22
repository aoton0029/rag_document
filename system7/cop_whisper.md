GitHub Copilot Chat Assistantです。以下、pyannote.audioで話者分離（ダイアリゼーション）し、faster_whisperで各話者区間を音声認識して時系列の話者付きテキストを作る手順と実用的なPythonスクリプト例を示します。実行環境（GPUの有無、wavの長さ）を教えていただければ、さらに最適化案を出します。

前提・インストール
- Hugging Faceのアクセストークンが必要（pyannoteの一部モデルがprivateまたは認証必要な場合）。
  export HUGGINGFACE_HUB_TOKEN="hf_..." を設定。
- 推奨パッケージ（例）:
  pip install pyannote.audio faster-whisper soundfile numpy torch

基本手順（要点）
1. pyannote.audioのPipelineでwavから話者区間（start, end, speakerラベル）を取得
2. 元wavを読み込み、各話者区間ごとに切り出す
3. 切り出した区間をfaster_whisperで文字起こし
4. 結果を時系列で結合して、speaker / start / end / text の形式で出力（必要ならSRTやCSVに変換）

サンプルコード（実用的なワンファイル例）
- このサンプルは soundfile を使って音声を読み書き、tempfileで区間毎に一時wavを作って faster_whisper に渡します。
- GPUがある場合は WhisperModel(..., device="cuda") にしてください。

```python
# 必要パッケージ: pyannote.audio, faster-whisper, soundfile, numpy, tempfile
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np
import tempfile
import os
import json

# 設定
WAV_PATH = "input.wav"
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")  # 必要に応じてセット
PYANNOTE_MODEL = "pyannote/speaker-diarization"     # 例。変える場合あり
WHISPER_MODEL = "small"                            # 例: tiny, base, small, medium, large
DEVICE = "cuda" if (os.environ.get("USE_CUDA")=="1") else "cpu"
LANG = "ja"  # 音声の言語（faster_whisperに渡す）

# 1) 話者分離（diarization）
pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=HF_TOKEN)
diarization = pipeline(WAV_PATH)  # diarizationはpyannote.core.Annotation

# 2) 元wav読み込み
audio, sr = sf.read(WAV_PATH, dtype="float32")
if audio.ndim > 1:
    # モノラルに変換（平均）
    audio = np.mean(audio, axis=1)

# 3) Whisperモデル準備
whisper = WhisperModel(WHISPER_MODEL, device=DEVICE)

results = []
# diarization.itertracks(yield_label=True) で (segment, track, label)
for segment, _, speaker in diarization.itertracks(yield_label=True):
    start = float(segment.start)
    end = float(segment.end)
    s_sample = int(start * sr)
    e_sample = int(end * sr)
    seg_audio = audio[s_sample:e_sample]

    # 小区間が短すぎる場合はスキップ（任意）
    if len(seg_audio) < 0.1 * sr:
        continue

    # 一時ファイルに書き出して transcribe に渡す方法（確実）
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, seg_audio, sr)

    # 4) 音声認識（区間ごと）
    segments, info = whisper.transcribe(tmp_path, language=LANG, task="transcribe", beam_size=5)
    # segments は faster_whisper の返すタイムスタンプ付き区間リスト（小区間が返ることがある）
    text = " ".join([seg.text.strip() for seg in segments]).strip()

    results.append({
        "speaker": speaker,
        "start": start,
        "end": end,
        "text": text
    })

    os.remove(tmp_path)

# 任意: 連続する同じ話者を結合する
merged = []
for r in results:
    if merged and merged[-1]["speaker"] == r["speaker"] and abs(merged[-1]["end"] - r["start"]) < 0.5:
        merged[-1]["end"] = r["end"]
        merged[-1]["text"] = merged[-1]["text"].rstrip() + " " + r["text"].lstrip()
    else:
        merged.append(r)

# 保存例(JSON)
with open("diarized_transcript.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

# SRT出力の簡易例
def to_srt_item(i, start, end, text):
    def fmt(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{i}\n{fmt(start)} --> {fmt(end)}\n[{text}]\n\n"

with open("diarized_transcript.srt", "w", encoding="utf-8") as srt:
    for i, r in enumerate(merged, start=1):
        srt.write(to_srt_item(i, r["start"], r["end"], f"{r['speaker']}: {r['text']}"))
```

注意点・改善案
- pyannoteのモデルはHugging Faceで認証が必要なことがあります。トークンを環境変数に設定してください。
- faster_whisperはGPUあると速いです。長時間音源（数十分〜）では区間ごとに transcribe を呼ぶとオーバーヘッドがあるため、複数区間をまとめてバッチで処理するほうが速くなる場合があります（ただし整合性に注意）。
- pyannoteの出力は重なり（オーバーラップ）を含むことがあります。重複区間の扱いはユースケースによって方針を決めてください（優先順位付け、重複分を分割して個別にASRなど）。
- 話者ラベルは "SPEAKER_00" のようなIDで返ります。特定人物名に紐付ける場合は別途話者同一性確認（スポークンID）や手動ラベリングが必要です。
- 音質が悪い/ノイズがある場合、前処理（ノイズ除去、音量正規化）で精度向上します。

追加で知りたいこと
- 実行環境（GPU有無、CUDAバージョン、wavの長さ）
- 出力形式（JSON / SRT / CSV / データベースへの保存 等）
- 1ファイルで良いか、フォルダ内の複数ファイルをバッチ処理したいか

希望を教えてください。必要なら上のスクリプトをあなたの環境用にカスタマイズして差し上げます。