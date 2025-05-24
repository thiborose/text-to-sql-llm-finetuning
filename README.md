
Goal: 
- Take any LLM, fine-tune it for SQL generation from natural language
- Publish the model and create a public inference endpoint
- Create a GUI, easily accessible

Assumptions: 
- As there are no requirements for the base pretrained model, I will take a general model. 
- The final LLM will be solely used for generating SQL, it does **not** matter if the LLM loses conversational abilities through the fine-tuning processs. 

Ideas: 
- Maybe I can focus on a set of data that assume the same SQL environment, that way I can test the output for real in a docker container 
- Also we should do a first check for out-of-topic input, maybe another LLM call or a classifier