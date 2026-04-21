# Tokenizer Visualizer

`tokenizer_visualizer` is a small web app for inspecting tokenizer boundaries with color-coded token output.


## What You Need

- Python with the server dependencies installed manually
- A tokenizer registry JSON file

Server dependencies:

```bash
pip install flask flask-cors transformers sentencepiece tokenizers
```

## Configuration

Copy the example tokenizer registry and edit it for your environment:

```bash
cp tokenizer_visualizer/tokenizers.example.json tokenizer_visualizer/tokenizers.local.json
```

Then point the server at it:

```bash
export TOKENIZER_VISUALIZER_CONFIG=/absolute/path/to/tokenizer_visualizer/tokenizers.local.json
```

Supported tokenizer sources:

- `hf`: Hugging Face tokenizer repo or local HF tokenizer directory
- `fast`: local `tokenizer.json` file or directory containing `tokenizer.json`
- `spm`: SentencePiece model file

Example entry:

```json
{
  "id": "llama3",
  "label": "Llama 3",
  "source": "hf",
  "path": "meta-llama/Meta-Llama-3-8B",
  "decode_mode": "byte_level"
}
```

Optional `decode_mode` values:

- `byte_level`: decode each token using byte-level reconstruction
- `token`: show raw token strings from `convert_ids_to_tokens`

## Run

From the repo root:

```bash
python3 tokenizer_visualizer/server/server.py
```

Default URL:

```text
http://127.0.0.1:5000
```

Optional environment variables:

- `TOKENIZER_VISUALIZER_CONFIG`: path to tokenizer registry JSON
- `TOKENIZER_VISUALIZER_HOST`: server host, default `127.0.0.1`
- `TOKENIZER_VISUALIZER_PORT`: server port, default `5000`

## API

### `GET /api/tokenizers`

Returns configured tokenizer metadata for the UI.

### `POST /api/tokenize`

Request:

```json
{
  "text": "Hello world",
  "tokenizerId": "llama3"
}
```

Response:

```json
{
  "tokens": ["Hello", " world"],
  "word_count": 2
}
```
