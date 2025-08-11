# LASR Probe Gen

```
git clone https://github.com/SamDower/LASR-probe-gen.git
cd LASR-probe-gen/
```


## 1. Sample and annotate dataset (LLM) instructions

- Uses GPT-4o API to label refusal behaviour
- Takes 4 minutes to do 10,000 samples

For off-policy labels:
```
uv run src/probe_gen/annotation/refusal_behaviour.py --path data/refusal/off_policy_raw.jsonl --num_samples 1000
```
For on-policy labels:
```
uv run src/probe_gen/annotation/refusal_behaviour.py --path data/refusal/on_policy_raw.jsonl --num_samples 1000 --outputs_hf NLie2/anthropic-refusal-activations
```


## 2. Get activations for dataset instructions

- Uses meta-llama/Llama-3.2-3B-Instruct to get actviations for on policy data
- Takes 1-2 hours to generate output activations for 1000 samples

For renting a GPU:
- Needs ??? GB disk
- Needs ??? GB RAM
- Needs 60 GB GPU

```
python get_activations.py \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --data "/rds/general/user/nk1924/home/LASR-probe-gen/data/refusal/anthropic_raw_apr_23.jsonl" \
  --out "/rds/general/user/nk1924/home/LASR-probe-gen/my_activations.pkl" \
  --batch-size 1 \
  --policy off_policy_other_model \
  --behaviour refusal
```


# Other
## ~~ Sample and annotate dataset (classifier) instructions~~

- Uses cais/HarmBench-Llama-2-13b-cls to label refusal
- Takes 5 minutes to do 1000 samples
- Specify main function for which file to take in

For renting a GPU:
- Needs 40 GB disk
- Needs 30-70 GB RAM
- Needs 100 GB GPU

```
uv run src/probe_gen/annotation/refusal_autograder.py
```
