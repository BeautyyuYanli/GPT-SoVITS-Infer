# GPT-SoVITS-Infer

This is the inference code of [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) that can be developer-friendly.

## Prepare the environment

As we all know, the dependencies of an AI project are always a mess. Here is how I prepare the environment for this project, by conda:

```
conda install python=3.10
conda install pytorch=2.1 torchvision torchaudio pytorch-lightning pytorch-cuda=12.1 -c pytorch -c nvidia 
conda install ffmpeg=6.1.1 -c conda-forge
```

You can also try to prepare the environment with cpu only options, which should work, but I have not tested it yet.

After the environment is ready, you can install the package by pip:

```
pip install GPT-SoVITS
```

I do not add the packages related to torch to the dependencies of GPT-SoVITS-Infer. Check if the environment is ready if things go wrong.

## Usage Example

Download the pretrained model from [HuggingFace](https://huggingface.co/lj1995/GPT-SoVITS) to `pretrained_models` folder (or any place you like):

```
git clone https://huggingface.co/lj1995/GPT-SoVITS pretrained_models
```

Create a `playground` folder (or any place you like) to save prompt audio and outputs:

```
mkdir playground
wget -P playground https://huggingface.co/datasets/BeautyyuYanli/sample_files/resolve/main/%E4%BD%A0%E5%A5%BD%20ChatGPT%EF%BC%8C%E8%AF%B7%E9%97%AE%E4%BD%A0%E7%9F%A5%E9%81%93%E4%B8%BA%E4%BB%80%E4%B9%88%E9%B2%81%E8%BF%85%E6%9A%B4%E6%89%93%E5%91%A8%E6%A0%91%E4%BA%BA%E5%90%97%EF%BC%9F.wav
```

The prompt audio should be about 3-10 seconds.

Generate the output:

```python
from scipy.io import wavfile

inference = GPTSoVITSInference(
    bert_path="pretrained_models/chinese-roberta-wwm-ext-large",
    cnhubert_base_path="pretrained_models/chinese-hubert-base",
)
inference.load_sovits("pretrained_models/s2G488k.pth")
inference.load_gpt(
    "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
)
prompt_text = "你好 ChatGPT，请问你知道为什么鲁迅暴打周树人吗？"
inference.set_prompt_audio(
    prompt_audio_path=f"playground/{prompt_text}.wav",
    prompt_text=prompt_text,
)

sample_rate, data = inference.get_tts_wav(
    text="鲁迅为什么暴打周树人？？？这是一个问题\n\n自古以来，文人相轻，鲁迅和周树人也不例外。鲁迅和周树人是中国现代文学史上的两位伟大作家，他们的文学成就都是不可磨灭的。但是，鲁迅和周树人之间的关系并不和谐，两人之间曾经发生过一次激烈的冲突，甚至还打了起来。那么，鲁迅为什么会暴打周树人呢？这是一个问题。  ",
)
wavfile.write(f"playground/output.wav", sample_rate, data)
```