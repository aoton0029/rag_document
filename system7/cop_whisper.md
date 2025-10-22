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



pyannote.audioとfaster_whisperを使用してwavファイルを話者分離付きの時系列テキストに変換するPythonコードを提示します。

## 必要なライブラリのインストール

```bash
pip install pyannote.audio faster-whisper torch torchaudio
```

## メインコード

```python
import torch
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
import wave
from datetime import timedelta

class SpeakerDiarization:
    def __init__(self, whisper_model_size="base", device="auto"):
        """
        初期化
        Args:
            whisper_model_size: Whisperモデルのサイズ (tiny, base, small, medium, large-v2など)
            device: 使用するデバイス (auto, cpu, cuda)
        """
        # Whisperモデルの初期化
        self.whisper_model = WhisperModel(whisper_model_size, device=device)
        
        # pyannote.audioの話者分離パイプラインの初期化
        # Hugging Face tokenが必要な場合があります
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="YOUR_HUGGINGFACE_TOKEN"  # 必要に応じて設定
        )
        
    def format_time(self, seconds):
        """秒を時:分:秒形式に変換"""
        return str(timedelta(seconds=int(seconds)))
    
    def transcribe_and_diarize(self, audio_file_path):
        """
        音声ファイルを話者分離付きで文字起こし
        Args:
            audio_file_path: wavファイルのパス
        Returns:
            話者別時系列テキストのリスト
        """
        print("音声認識を開始...")
        
        # Whisperで音声認識
        segments, info = self.whisper_model.transcribe(
            audio_file_path,
            beam_size=5,
            language="ja"  # 日本語を指定
        )
        
        print(f"検出された言語: {info.language} (信頼度: {info.language_probability:.2f})")
        
        # 音声認識結果を辞書形式で保存
        transcription_segments = []
        for segment in segments:
            transcription_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text
            })
        
        print("話者分離を開始...")
        
        # pyannote.audioで話者分離
        diarization = self.diarization_pipeline(audio_file_path)
        
        # 話者分離結果を辞書形式で保存
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        print("音声認識結果と話者分離結果を統合...")
        
        # 音声認識結果と話者分離結果を統合
        final_results = self.merge_transcription_and_diarization(
            transcription_segments, speaker_segments
        )
        
        return final_results
    
    def merge_transcription_and_diarization(self, transcription_segments, speaker_segments):
        """音声認識結果と話者分離結果を統合"""
        final_results = []
        
        for trans_segment in transcription_segments:
            trans_start = trans_segment['start']
            trans_end = trans_segment['end']
            trans_text = trans_segment['text']
            
            # この音声認識セグメントと最も重複する話者を見つける
            best_speaker = "不明"
            max_overlap = 0
            
            for speaker_segment in speaker_segments:
                speaker_start = speaker_segment['start']
                speaker_end = speaker_segment['end']
                speaker_label = speaker_segment['speaker']
                
                # 重複部分の計算
                overlap_start = max(trans_start, speaker_start)
                overlap_end = min(trans_end, speaker_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = speaker_label
            
            final_results.append({
                'start_time': trans_start,
                'end_time': trans_end,
                'speaker': best_speaker,
                'text': trans_text.strip()
            })
        
        return final_results
    
    def save_results(self, results, output_file="transcription_with_speakers.txt"):
        """結果をファイルに保存"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("話者分離付き音声認識結果\n")
            f.write("=" * 50 + "\n\n")
            
            for result in results:
                start_time = self.format_time(result['start_time'])
                end_time = self.format_time(result['end_time'])
                speaker = result['speaker']
                text = result['text']
                
                f.write(f"[{start_time} - {end_time}] {speaker}: {text}\n")
        
        print(f"結果を {output_file} に保存しました。")
    
    def print_results(self, results):
        """結果をコンソールに出力"""
        print("\n話者分離付き音声認識結果:")
        print("=" * 50)
        
        for result in results:
            start_time = self.format_time(result['start_time'])
            end_time = self.format_time(result['end_time'])
            speaker = result['speaker']
            text = result['text']
            
            print(f"[{start_time} - {end_time}] {speaker}: {text}")

def main():
    """メイン関数"""
    # 音声ファイルのパスを指定
    audio_file = "input_audio.wav"  # ここにwavファイルのパスを指定
    
    # 話者分離オブジェクトの作成
    diarizer = SpeakerDiarization(
        whisper_model_size="base",  # モデルサイズを調整可能
        device="auto"  # GPUが利用可能な場合は自動的に使用
    )
    
    try:
        # 音声認識と話者分離を実行
        results = diarizer.transcribe_and_diarize(audio_file)
        
        # 結果を画面に表示
        diarizer.print_results(results)
        
        # 結果をファイルに保存
        diarizer.save_results(results, "output_transcription.txt")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
```

## 使用方法

1. **Hugging Face トークンの設定**（pyannote.audioで必要な場合）:
   ```python
   # Hugging Faceアカウントでトークンを取得し、コード内で設定
   use_auth_token="YOUR_HUGGINGFACE_TOKEN"
   ```

2. **音声ファイルの準備**:
   - `input_audio.wav`を実際のwavファイルパスに変更

3. **実行**:
   ```bash
   python speaker_diarization.py
   ```

## 出力例

```
[0:00:00 - 0:00:05] SPEAKER_00: こんにちは、今日はお忙しい中お時間をいただき、ありがとうございます。
[0:00:06 - 0:00:10] SPEAKER_01: こちらこそ、よろしくお願いします。
[0:00:11 - 0:00:15] SPEAKER_00: それでは、早速始めさせていただきます。
```

## 設定オプション

- **Whisperモデルサイズ**: `tiny`, `base`, `small`, `medium`, `large-v2`
- **デバイス**: `auto`, `cpu`, `cuda`（GPU使用時）
- **言語**: `language="ja"`で日本語を指定

このコードで、wavファイルから話者分離付きの時系列テキストを生成できます。