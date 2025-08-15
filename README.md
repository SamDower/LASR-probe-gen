# LASR Probe Gen

```
git clone https://github.com/SamDower/LASR-probe-gen.git
cd LASR-probe-gen/
```


## 1. Sample and annotate dataset

- Uses GPT-4o API to label refusal behaviour
- Takes 4 minutes to do 10,000 samples
- Hardware requirements: None
- Make sure you have done 'export OPENAI_API_KEY=<key>'

For sampling a new dataset with off-policy outputs:
```
uv run src/probe_gen/annotation/refusal_behaviour.py --out_path data/refusal/off_policy_raw.jsonl --num_samples 1000 --do_label True --do_subsample True
```
For labelling an existing dataset with on-policy outputs:
```
uv run src/probe_gen/annotation/refusal_behaviour.py --out_path data/refusal/on_policy_raw_20k.jsonl --in_path data/refusal/on_policy_unlabelled_20k.jsonl --do_label True --do_subsample True
```
Where --do_label and --do_subsample are True by default.


## 2. Get on policy outputs for dataset

- Uses meta-llama/Llama-3.2-3B-Instruct to get outputs
- Takes 4 minutes to generate outputs for 20,000 samples

Hardware requirements: 10 GB GPU

```
uv run python scripts/get_outputs.py --data data/refusal/off_policy_raw_20k.jsonl --out on_policy_raw_20k.pkl --batch-size 200 --sample 0 --behaviour refusal

```


## 3. Get activations for dataset

- Uses meta-llama/Llama-3.2-3B-Instruct to get actviations for on policy data
- Takes 1-2 hours to generate output activations for 1000 samples
- Hardware requirements: 60 GB GPU, ??? GB RAM, ??? GB Disk

```
python get_activations.py \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --data "/rds/general/user/nk1924/home/LASR-probe-gen/data/refusal/anthropic_raw_apr_23.jsonl" \
  --out "/rds/general/user/nk1924/home/LASR-probe-gen/my_activations.pkl" \
  --batch-size 1 \
  --policy off_policy_other_model \
  --behaviour refusal
```

## 4. Train probes on activations dataset
- Currently just using notebooks/TrainProbe.ipynb and running cells
- Hardware requirements: 0 GB GPU, ??? GB RAM, ??? GB Disk


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
Only now open the notebook and go to Kernel → Change Kernel → Python (uv).


## Use autograder on dataset (unused)

- Uses cais/HarmBench-Llama-2-13b-cls to label refusal
- Takes 5 minutes to do 1000 samples
- Hardware requirements: 40 GB disk, 30-70 GB RAM, 100 GB GPU

```
uv run src/probe_gen/annotation/refusal_autograder.py
```
