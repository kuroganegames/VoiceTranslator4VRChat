# coding: utf-8
"""
Voice Translator for VRChat (refactored)
"""

import os
from abc import ABC, abstractmethod
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import soundfile as sf
import speech_recognition as sr
import torch
from pythonosc import udp_client
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

# --------------------------------------------------
# ★ 設定用定数  ★
# --------------------------------------------------
MIC_DEVICE_INDEX         = 2
MIC_SAMPLE_RATE          = 16_000

VRC_IP                   = "192.168.50.118"           # VRChat OSC サーバの IP
VRC_PORT                 = 9000                      # VRChat OSC サーバのポート
CHATBOX_INPUT_ADDRESS    = "/chatbox/input"          # OSC アドレス

MODEL_ID_WHISPER         = "kotoba-tech/kotoba-whisper-v2.0"
MODEL_ID_TRANSLATION     = "facebook/nllb-200-distilled-600M"

CACHE_PATH               = r"G:\cache"               # 🤗 Transformers のキャッシュ先

REMOVE_WORDS             = ["ごめん"]                # 無音誤認識ワードの除外
TRANSLATION_MAX_LENGTH   = 30                        # 翻訳文の最大トークン長
LOG_HISTORY_SIZE         = 3                         # コンソールに表示する履歴数
DEVICE                   = "cuda" if torch.cuda.is_available() else "cpu"

# OpenAI API設定（OpenAI翻訳エンジン使用時）
OPENAI_API_KEY           = os.getenv("OPENAI_API_KEY", "")  # 環境変数から取得
OPENAI_MODEL             = "gpt-3.5-turbo"                   # 使用するモデル
OPENAI_API_BASE          = None                              # カスタムエンドポイント（None = デフォルト）

# 翻訳エンジンの選択: "nllb" or "openai"
TRANSLATION_ENGINE       = "nllb"

# --------------------------------------------------
# 以降はコード本体
# --------------------------------------------------


def clear_console() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def print_banner(
    mode: str,
    history: List[str],
    translator_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    画面をクリアしてバナーと履歴を描画
    """
    if translator_info is None:
        translator_info = {}
    clear_console()
    history_lines = "\n".join(
        f"log{idx}:\t{line}"
        for idx, line in zip(
            range(LOG_HISTORY_SIZE, 0, -1), reversed(history[-LOG_HISTORY_SIZE :])
        )
    )
    print(
        f"""----------
Voice Translator for VRChat
----------
[MIC]
\tSELECTED INDEX:\t{MIC_DEVICE_INDEX}
\tSAMPLING_RATE:\t{MIC_SAMPLE_RATE}
[VRCHAT OCS CONNECTION]
\tIP:\t{VRC_IP}
\tPORT:\t{VRC_PORT}
[ML MODEL]
\tVOICE RECOGNITION MODEL:\t{MODEL_ID_WHISPER}
\tTRANSLATION ENGINE:\t{translator_info.get('name', 'Unknown')}
\tTRANSLATION MODEL:\t{translator_info.get('model', 'Unknown')}
\tDIR_MODEL_SAVE:\t{CACHE_PATH}
[REMOVE WORDS]
\t:\t{", ".join(REMOVE_WORDS)}
----------
MODE: {mode}
---
[HISTORY]
{history_lines}
----------""",
        flush=True,
    )


def send_to_vrchat_chatbox(message: str) -> None:
    """
    VRChat ChatBox にテキストを送信
    """
    client = udp_client.SimpleUDPClient(VRC_IP, VRC_PORT)
    client.send_message(CHATBOX_INPUT_ADDRESS, [message, True])


class TranslatorEngine(ABC):
    """翻訳エンジンの抽象基底クラス"""
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """翻訳エンジンの初期化"""
        pass
    
    @abstractmethod
    def translate(self, text: str, source_lang: str = "ja", target_lang: str = "en") -> str:
        """テキストを翻訳"""
        pass
    
    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """エンジン情報を取得"""
        pass


class NLLBTranslator(TranslatorEngine):
    """Facebook NLLB翻訳エンジン"""
    
    def initialize(self, **kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID_TRANSLATION,
            token=True,
            src_lang="jpn_Jpan",
            cache_dir=CACHE_PATH,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID_TRANSLATION,
            token=True,
            cache_dir=CACHE_PATH,
        )
    
    def translate(self, text: str, source_lang: str = "ja", target_lang: str = "en") -> str:
        inputs = self.tokenizer(text, return_tensors="pt")
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=TRANSLATION_MAX_LENGTH,
        )
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "NLLB",
            "model": MODEL_ID_TRANSLATION,
            "type": "local"
        }


class OpenAITranslator(TranslatorEngine):
    """OpenAI API互換翻訳エンジン"""
    
    def initialize(self, **kwargs) -> None:
        try:
            import openai
            self.openai = openai
            
            # API設定
            if OPENAI_API_KEY:
                self.client = openai.OpenAI(
                    api_key=OPENAI_API_KEY,
                    base_url=OPENAI_API_BASE
                )
            else:
                raise ValueError("OPENAI_API_KEY is not set")
                
            self.model = kwargs.get("model", OPENAI_MODEL)
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
    
    def translate(self, text: str, source_lang: str = "ja", target_lang: str = "en") -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"Translate the following {source_lang} text to {target_lang}. Provide only the translation without any explanation."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=TRANSLATION_MAX_LENGTH * 4  # トークン数の概算
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] OpenAI API error: {e}")
            return "[Translation Error]"
    
    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI",
            "model": self.model,
            "type": "api",
            "base_url": OPENAI_API_BASE or "default"
        }


class VoiceTranslator:
    """音声→テキスト→翻訳→VRChat 送信のワークフローを管理"""

    def __init__(self, translation_engine: Optional[str] = None) -> None:
        self.recognizer = sr.Recognizer()
        self.processor = WhisperProcessor.from_pretrained(
            MODEL_ID_WHISPER, cache_dir=CACHE_PATH
        )
        self.model_whisper = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID_WHISPER, cache_dir=CACHE_PATH
        ).to(DEVICE)

        # 翻訳エンジンの選択と初期化
        engine_type = translation_engine or TRANSLATION_ENGINE
        self.translator = self._create_translator(engine_type)
        self.translator.initialize()
        
        self.history: List[str] = []
    
    def _create_translator(self, engine_type: str) -> TranslatorEngine:
        """指定されたタイプの翻訳エンジンを作成"""
        if engine_type.lower() == "nllb":
            return NLLBTranslator()
        elif engine_type.lower() == "openai":
            return OpenAITranslator()
        else:
            raise ValueError(f"Unknown translation engine: {engine_type}")

    # ---------- 音声認識 → 日本語文字列 ---------- #
    def recognize_speech(self, wav_bytes: bytes) -> str:
        wav_stream = BytesIO(wav_bytes)
        audio_array, sampling_rate = sf.read(wav_stream)

        input_features = self.processor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features.to(DEVICE)

        generated_ids = self.model_whisper.generate(input_features)
        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

    # ---------- 日本語 → 英語翻訳 ---------- #
    def translate_to_en(self, text: str) -> str:
        return self.translator.translate(text, source_lang="ja", target_lang="en")

    # ---------- 翻訳＋フィルタ ---------- #
    def filter_and_translate(self, jp_text: str) -> Tuple[str, bool]:
        if jp_text in REMOVE_WORDS:
            return "", False

        en_text = self.translate_to_en(jp_text)
        return f"{jp_text}\n{en_text}", True

    # ---------- メインループ ---------- #
    def run(self) -> None:
        mode = "起動中..."
        translator_info = self.translator.get_engine_info()
        print_banner(mode, self.history, translator_info)

        try:
            while True:
                mode = "なにか話してください"
                print_banner(mode, self.history, translator_info)

                # 録音
                with sr.Microphone(
                    device_index=MIC_DEVICE_INDEX, sample_rate=MIC_SAMPLE_RATE
                ) as source:
                    audio_data = self.recognizer.listen(source)

                # 音声認識
                mode = "音声認識中..."
                print_banner(mode, self.history, translator_info)
                jp_text = self.recognize_speech(audio_data.get_wav_data())

                # 翻訳
                mode = "翻訳中..."
                print_banner(mode, self.history, translator_info)
                text_to_send, should_send = self.filter_and_translate(jp_text)

                # 送信 & ログ
                if should_send:
                    send_to_vrchat_chatbox(text_to_send)
                    self.history.append(text_to_send.replace("\n", "\t"))

        except KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt — exiting gracefully.")


def main() -> None:
    # コマンドライン引数やconfig から翻訳エンジンを選択可能
    # 例: translator = VoiceTranslator("openai")  # OpenAI APIを使用
    translator = VoiceTranslator()  # デフォルト設定を使用
    translator.run()


if __name__ == "__main__":
    main()