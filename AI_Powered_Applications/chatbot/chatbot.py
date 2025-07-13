from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

while True:
    input_text = input("> ")
    if input_text.lower() in ["exit", "quit"]:
        break
    history_string = "\n".join(conversation_history)
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Response:", response)
    conversation_history.append(input_text)
    conversation_history.append(response)
