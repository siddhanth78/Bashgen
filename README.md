## BashGen 0.1

- Made for Mac OS
- Bash command generation assistant

## Setup

```
git clone https://github.com/siddhanth78/Bashgen`
cd BashGen
pip install -r requirements.txt
```

- Download Ollama from here: https://ollama.com/download
- Make sure Ollama server is up when using BashGen

- Download LLaMA 3.1 8B

`ollama pull llama3.1:8b`

- Create custom model with tuned parameters

`ollama create myllama3_1 --file myllama3_1.modelfile`

- Run BashGen 0.1

`python3 bashgen0_1.py`
