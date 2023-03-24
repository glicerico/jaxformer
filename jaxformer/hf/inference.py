from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def predict(model, prompt, dev):
    """
    Predict the next tokens after the prompt, from given model
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    device = torch.device(dev)
    tokenized_sent = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        # input_ids = data[0].to(device)
        output = model.generate(tokenized_sent["input_ids"].to(device))
        print(f"Result: {tokenizer.decode(output[0])}")
# Run inference on fine-tuned models


model_type = "Salesforce/codegen-350M-multi"
saved_model = AutoModelForCausalLM.from_pretrained(model_type)
saved_model.load_state_dict(torch.load("pytorch_model.bin"))

saved_model.eval()
test_sent = "Sentence: The boy is not tired today . AMR: "
print("Saved model")
predict(saved_model, test_sent, "cpu")
