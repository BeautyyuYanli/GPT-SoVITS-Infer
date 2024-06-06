from gpt_sovits.infer.worker import GPTSoVITSInference
from pydantic import BaseModel
from typing import List, Tuple, Optional
from pathlib import Path
from threading import Lock
import os, sys


class ConfigData(BaseModel):
    models: List[str]
    prompts: List[str]


class GPTSoVITSInferenceSimple:
    config_data_base: Path
    config_data: ConfigData
    working_config: Tuple[str, str]

    inference_worker: GPTSoVITSInference
    inference_worker_lock: Lock

    def __init__(
        self,
        config_data_base: str,
        inference_worker_and_lock: Tuple[GPTSoVITSInference, Lock],
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
    ):
        self.config_data_base = Path(config_data_base)
        model_files = os.listdir(str(self.config_data_base / "models"))
        models = [
            model_file.split(".")[0]
            for model_file in model_files
            if model_file.endswith(".pth")
            and model_file.replace(".pth", ".ckpt") in model_files
        ]
        prompt_files = os.listdir(str(self.config_data_base / "prompts"))
        prompts = [
            prompt_file.split(".")[0]
            for prompt_file in prompt_files
            if prompt_file.endswith(".txt")
            and prompt_file.replace(".txt", ".wav") in prompt_files
        ]
        self.config_data = ConfigData(models=models, prompts=prompts)

        self.inference_worker = inference_worker_and_lock[0]
        self.inference_worker_lock = inference_worker_and_lock[1]
        self.working_config = (
            model_name if model_name else models[0],
            prompt_name if prompt_name else prompts[0],
        )
        self._load_model(self.working_config[0])
        self._load_prompt(self.working_config[1])

    def _load_model(self, model_name: str):
        self.inference_worker.load_sovits(
            str(self.config_data_base / "models" / f"{model_name}.pth")
        )
        self.inference_worker.load_gpt(
            str(self.config_data_base / "models" / f"{model_name}.ckpt")
        )

    def _load_prompt(self, prompt_name: str):
        with open(self.config_data_base / "prompts" / f"{prompt_name}.txt", "r") as f:
            prompt_text = f.read().strip()
        self.inference_worker.set_prompt_audio(
            prompt_text=prompt_text,
            prompt_audio_path=str(
                self.config_data_base / "prompts" / f"{prompt_name}.wav"
            ),
        )

    def generate(
        self,
        text: str,
        text_language="auto",
        top_k=5,
        top_p=1,
        temperature=1,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
    ):
        config = (
            model_name if model_name else self.working_config[0],
            prompt_name if prompt_name else self.working_config[1],
        )
        with self.inference_worker_lock:
            if config[0] != self.working_config[0]:
                self._load_model(config[0])
            if config[1] != self.working_config[1]:
                self._load_prompt(config[1])
            self.working_config = config
            return self.inference_worker.get_tts_wav(
                text=text,
                text_language=text_language,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )

    def generate_stream(
        self,
        text: str,
        text_language="auto",
        top_k=5,
        top_p=1,
        temperature=1,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
    ):
        config = (
            model_name if model_name else self.working_config[0],
            prompt_name if prompt_name else self.working_config[1],
        )
        with self.inference_worker_lock:
            if config[0] != self.working_config[0]:
                self._load_model(config[0])
            if config[1] != self.working_config[1]:
                self._load_prompt(config[1])
            self.working_config = config
            for thing in self.inference_worker.get_tts_wav_stream(
                text=text,
                text_language=text_language,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            ):
                yield thing

        return None


if __name__ == "__main__":
    from scipy.io import wavfile

    inference_worker_and_lock = (
        GPTSoVITSInference(
            bert_path="pretrained_models/chinese-roberta-wwm-ext-large",
            cnhubert_base_path="pretrained_models/chinese-hubert-base",
        ),
        Lock(),
    )

    api = GPTSoVITSInferenceSimple(
        config_data_base="config_data",
        inference_worker_and_lock=inference_worker_and_lock,
    )
    for idx, (sr, data) in enumerate(
        api.generate_stream("鲁迅为什么暴打周树人？这是一个问题")
    ):
        wavfile.write(f"playground/output/output{idx}.wav", sr, data)
    sr, data = api.generate("鲁迅为什么暴打周树人？这是一个问题")
    wavfile.write("playground/output.wav", sr, data)
