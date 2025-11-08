import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def run_inference():
    """
    Loads the base model, composes the domain and task adapters,
    and runs inference on a sample prompt.
    """
    model_name = "gpt2"
    domain_adapter_path = "models/domain_adapter"
    task_adapter_path = "models/task_adapter"

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and compose the domain adapter
    model = PeftModel.from_pretrained(model, domain_adapter_path, adapter_name="domain_adapter")
    print("Domain adapter loaded.")

    # Load and compose the task adapter
    model.load_adapter(task_adapter_path, adapter_name="task_adapter")
    print("Task adapter loaded.")

    # By loading the task adapter on top of the domain adapter, their weights are effectively composed.
    # No need to call `set_adapter` unless you want to switch between them.
    print("Domain and task adapters composed.")

    # Create a sample prompt
    prompt = "Context: The capital of France is Paris. The Eiffel Tower is a famous landmark in Paris.\n\nQuestion: What is the capital of France?\n\nAnswer:"

    # Tokenize the prompt and generate a response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n--- Inference ---")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("-----------------\n")

if __name__ == "__main__":
    run_inference()
