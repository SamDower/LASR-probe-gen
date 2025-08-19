# LASR Probe Gen
See notebooks/DataPipeline.ipynb to get the datasets of inputs, outputs, labels and activations. \
See notebooks/TrainProbes.ipynb to train and evaluate probes on the datasets.
```
git clone https://github.com/SamDower/LASR-probe-gen.git
cd LASR-probe-gen/
```

# Full Dataset Pipeline
## 1. Sample inputs dataset
- Samples dataset from hugging face to jsonl file
- For refusal behaviour can save time by doing labelling and subsamplling of off policy outpus at the same time (set to 'yes')

```uv run scripts/get_dataset_labelled.py --behaviour refusal --out_path data/refusal/claude_outputs.jsonl --num_samples 1000 --do_label no --do_subsample no```

## 2. Generate outputs dataset (on-policy)
- Uses LLM (Llama-3.2-3B-Instruct default) to generate outputs for inputs dataset
- Takes 10 minutes to do 5k samples with 200 batch size
- Hardware requirements: high GPU, low RAM, low disk
- Make sure you have done 'export HF_TOKEN=<key>' or just paste it here but cant push to git

```uv run scripts/get_outputs.py --data data/refusal/claude_outputs.jsonl --out llama_3b_outputs.jsonl --batch-size 200 --sample 0  --policy on_policy --behaviour refusal --save-increment -1```

## 3. Label and balance dataset
- Uses GPT-4o API to label refusal behaviour
- Takes 4 minutes to do 10k samples
- Hardware requirements: None
- Make sure you have done 'export OPENAI_API_KEY=<key>'

```uv run scripts/get_dataset_labelled.py --behaviour refusal --out_path data/refusal/llama_3b_raw.jsonl --in_path data/refusal/llama_3b_outputs.jsonl --do_label True --do_subsample True```

## 4. Get activations dataset
- Uses LLM (Llama-3.2-3B-Instruct default) to get actviations for datasets
- Takes 10 minutes to generate output activations for 5k samples with 200 batch size
- Hardware requirements: high GPU, super high (150 GB) RAM, super high (150 GB) Disk
- Make sure you have done 'export HF_TOKEN=<key>' or just paste it here but cant push to git

```uv run scripts/get_activations.py --model "meta-llama/Llama-3.2-3B-Instruct" --data data/refusal/llama_3b_balanced_5k.jsonl --batch-size 1 --layers all --save-increment -1```

Upload the activations to hugging face using notebooks/DataPipeline.ipynb

# Other
## Connect vscode to vast.ai
Open a terminal that isnt WSL (e.g. Windows/ Mac/ Native Linux) and run this command while just pressing enter for each option, to set up a private and public ssh key:
```
ssh-keygen -t rsa
```
Then copy the contents of the public key file (e.g. at "C:\Users\<username_here>\.ssh\id_rsa.pub") and add it to vast.ai at https://cloud.vast.ai/manage-keys/ clicking 'SSH Keys' tab at the top.\
Then create a GPU instance and click on 'Terminal Connection Options' near the 'Open' button to get the ssh command, which you then add to to specify your private ssh key location to. It should look like this:
```
 ssh -p 55327 root@199.126.134.31 -L 8080:localhost:8080 -i ~\.ssh\id_rsa
 ```
Then open vscode without WSL connection and select 'Connect to host...' and paste the ssh command in. \
It might ask you to choose a config to save the ssh instruction to, in which case do that and then redo 'Connect to host...' but this time just select the ssh IP from the list instead of pasting the command. \
You should be connected. Now to open the workspace folder, click File → Open folder instead of using the explorer menu. \
To move files over, you can git clone or pull from google drive or scp from local files to the instance, for example:
```
scp -P 55327 local_file.py root@199.126.134.31:/workspace
```


## Run notebooks in vscode connected to vast.ai
Same as below but it might ask you to install vscode extensions for python and jupyter first and then click 'Select kernel' in the top right. Also, it may be possible to avoid creating a new kernel if selecting the 'probe-gen .venv' works.


## Run notebooks in vast.ai browser
When not running individual python scripts and want to use JupyterLabs, need to set up a new kernel in the Jupyter terminal:
```
uv sync
uv run python -m ipykernel install --user --name=uv-env --display-name "Python (uv)"
```
Now open the notebook and go to Kernel → Change Kernel → Python (uv). You may need to press the refresh button if it is not showing up.
