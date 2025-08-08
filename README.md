# LASR Probe Gen

```
git clone https://github.com/SamDower/LASR-probe-gen.git
cd LASR-probe-gen/
```

## Autograder (classifier) instructions

- Uses cais/HarmBench-Llama-2-13b-cls to label refusal
- Takes 5 minutes to do 1000 samples

For renting a GPU:
- Needs like 40 GB disk
- Needs like 30-70 GB RAM
- Needs like 100GB GPU

```
uv run src/probe_gen/autograder.py
```


## Autograder (LLM) instructions

- Uses GPT-4o API to label refusal
- Currently 

```
uv run src/probe_gen/harmbench_autograder.py
```