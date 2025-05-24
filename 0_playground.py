import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from dotenv import load_dotenv
    from huggingface_hub import login as hf_login

    return hf_login, load_dotenv, os


@app.cell
def _(hf_login, load_dotenv, os):
    load_dotenv()

    hf_login(
        token=os.getenv("HF_TOKEN")
    )
    return


@app.cell
def _():
    # Import necessary libraries
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer, setup_chat_format
    import torch

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load the model and tokenizer
    model_name = "HuggingFaceTB/SmolLM2-135M"
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    # Set up the chat format
    model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    # Set our name for the finetune to be saved &/ uploaded to
    finetune_name = "SmolLM2-FT-MyDataset"
    finetune_tags = ["smol-course", "module_1"]
    return (
        SFTConfig,
        SFTTrainer,
        device,
        finetune_name,
        load_dataset,
        model,
        tokenizer,
    )


@app.cell
def _(tokenizer):
    # Let's test the base model before training
    prompt = "Hello?"

    # Format with template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    formatted_prompt
    return (formatted_prompt,)


@app.cell
def _(device, formatted_prompt, tokenizer):
    # tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    inputs
    return (inputs,)


@app.cell
def _(inputs, model, tokenizer):
    outputs = model.generate(**inputs, max_new_tokens=100)
    print("Before training:")
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded_output)
    return (decoded_output,)


@app.cell
def _(decoded_output):
    decoded_output
    return


@app.cell
def _(load_dataset):
    ds = load_dataset(path="HuggingFaceTB/smoltalk", name="everyday-conversations")
    return (ds,)


@app.cell
def _(SFTConfig, finetune_name):
    	# Configure the SFTTrainer
    sft_config = SFTConfig(
        output_dir="./sft_output",
        max_steps=1000,  # Adjust based on dataset size and desired training duration
        per_device_train_batch_size=4,  # Set according to your GPU memory capacity
        learning_rate=5e-5,  # Common starting point for fine-tuning
        logging_steps=10,  # Frequency of logging training metrics
        save_steps=100,  # Frequency of saving model checkpoints
        # evaluation_strategy="steps",  # Evaluate the model at regular intervals
        eval_steps=50,  # Frequency of evaluation
        hub_model_id=finetune_name,  # Set a unique name for your model
    )
    return (sft_config,)


@app.cell
def _(SFTTrainer, ds, model, sft_config, tokenizer):
    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        processing_class=tokenizer,
        eval_dataset=ds["test"],
    )
    return (trainer,)


@app.cell
def _(finetune_name, trainer):
    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(f"./{finetune_name}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
