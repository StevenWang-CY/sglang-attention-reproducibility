curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-30B-A3B-Instruct",
    "temperature": 0.0,
    "max_tokens": 32,
    "messages": [
      {"role": "system", "content": [{"type":"text","text":"You are a helpful assistant."}]},
      {"role": "user", "content": [{"type":"text","text":"Say hello."}]}
    ]
  }'
