curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen3-VL-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": "Hello, how are you?"}]
      }'
      
