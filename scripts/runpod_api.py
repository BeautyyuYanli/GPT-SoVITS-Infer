import io
import os
import dotenv
import runpod
import base64
import gzip
import numpy as np
from pydantic import BaseModel, Field
from pydub import AudioSegment

from gpt_sovits.infer.interface import GPTSoVITSInterfaceSimple

dotenv.load_dotenv()


class InputType(BaseModel):
    the_model_name: str = Field(alias="model_name")
    prompt_name: str
    text: str
    text_language: str = "auto"
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    use_stream: bool = False


interface = GPTSoVITSInterfaceSimple(
    config_data_base=os.getenv("GPT_SOVITS_CONFIG", ""),
)


def _encode_string(sr: int, data: np.ndarray) -> bytes:
    audio = AudioSegment(
        data=data.tobytes(),
        sample_width=data.dtype.itemsize,
        frame_rate=sr,
        channels=1,
    )
    with io.BytesIO() as in_memory_file:
        audio.export(
            in_memory_file,
            format="ogg",
            codec="opus",
            parameters=["-strict", "-2", "-vbr", "3"],
        )
        bytes_data = in_memory_file.read()
        print(len(bytes_data))
        return base64.standard_b64encode(gzip.compress(bytes_data))


def handler_fn(job):
    input_val = InputType.model_validate(job["input"])
    if input_val.use_stream:
        for sr, data in interface.generate_stream(
            input_val.the_model_name,
            input_val.prompt_name,
            input_val.text,
            text_language=input_val.text_language,
            top_k=input_val.top_k,
            top_p=input_val.top_p,
            temperature=input_val.temperature,
        ):
            yield {
                "data": _encode_string(sr, data),
            }
    else:
        sr, data = interface.generate(
            input_val.the_model_name,
            input_val.prompt_name,
            input_val.text,
            text_language=input_val.text_language,
            top_k=input_val.top_k,
            top_p=input_val.top_p,
            temperature=input_val.temperature,
        )
        yield {
            "data": _encode_string(sr, data),
        }


runpod.serverless.start(
    {
        "handler": handler_fn,
        "return_aggregate_stream": True,
        "concurrency_modifier": lambda x: 1,
    }
)
