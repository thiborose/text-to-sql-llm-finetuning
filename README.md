## Fine-Tuning the LLM 

I decided to fine-tune [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) because: 
- It is relatively small, with 135M parameters and a total size of less than 1GB. This will give me more flexibility to choose my training environment, the training will be quicker, and it will allow to run inference on a CPU.
- It has been trained for general language and performs poorly on SQL generation, so that the impact of the fine-tuning will be clearer. 

For the fine-tuning tasks, I made the following assumptions: 
- The only task of interest is SQL generation, therefore, it does **not** matter if the LLM loses other abilities through the fine-tuning processs, such as general language capabilities.
- I will **not** include contextual information into the training such as the existing SQL environment (tables, columns), or explanations of the output SQL query. Although this would be an interesting approach to explore to create a more robust model, I decided to keep it simple due to the limited time available. 

Fine-Tuning Process
- I used a Google Colab-notebook with a T4 GPU runtime
- I used the training part of the dataset [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql), and specifically the columns `sql` and `sql_prompt`
- Use the library `trl`
- Used the following parameters:
    - `max_steps`: `3000`
    - `per_device_train_batch_size`: `2`
    - `gradient_accumulation_steps`: `8`
    - `learning_rate`: `2e-5`
    - `save_steps`: `500`
    - `eval_steps`: `500`

I published the fine-tuned model on hugginface at [`thiborose/SmolLM2-FT-SQL`](https://huggingface.co/thiborose/SmolLM2-FT-SQL)

## Evaluating the Fine-Tuned LLM
<!-- TODO
Metrics: 
- Is valid sql code (check syntax, can be done autmatically)
- Does it correspond to the input (use another LLM, e.g. gpt instance)

-->

```
az cognitiveservices account create   --name azoai   --resource-group nl2sql   --kind OpenAI   --sku S0   --location westeurope   --yes

az cognitiveservices account deployment create \
  --name azoai \
  --resource-group nl2sql \
  --deployment-name o4-mini \
  --model-name gpt-o4-mini \
  --model-version 2025-04-16 \
  --model-format OpenAI
```

## Deploying the LLM

I created a simple streamlit app hosted as a container instance in Azure. The web app is available at ""

Deployment steps:
```
az login

az group create -n nl2sql -l northeurope

az acr create --resource-group nl2sql --name nl2sqlregistry --sku Basic

az acr login --name nl2sqlregistry.azurecr.io

docker-buildx build --platform linux/amd64 -t nl2sqlregistry.azurecr.io/nl2sql:latest .

docker push nl2sqlregistry.azurecr.io/nl2sql:latest
```

