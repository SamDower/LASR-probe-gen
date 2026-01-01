"""
Standalone Probe Experiment Script

Complete pipeline for probe training and evaluation without dependencies on probe_gen.
"""

import asyncio
import json
import os
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from openai import AsyncOpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

# Dataset configuration
HF_DATASET_NAME = "your_dataset_name"  # Replace with your HuggingFace dataset
HF_DATASET_SPLIT = "train"
HF_DATASET_COLUMN_INPUT = "text"  # Column name for input text

# Model configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with your model
GENERATION_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 200
BATCH_SIZE = 8

# GPT Classification configuration
GPT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """You are classifying model outputs. For each input-output pair, determine if the output exhibits the target behavior.

Respond with a JSON object containing:
- "answer": an integer from 1-10 where 1 means strong positive, 10 means strong negative
- "reason": a brief explanation
- "confidence": an integer from 1-10 indicating your confidence"""

NUM_BALANCED = 5000  # Total balanced samples (2500 pos + 2500 neg)
CONFIDENCE_THRESHOLD = 7
POSITIVE_THRESHOLD = 3  # Answer <= 3 is positive
NEGATIVE_THRESHOLD = 8  # Answer >= 8 is negative

# Probe configuration
LAYER = 12  # Layer to extract activations from
TRAIN_SIZE = 3500
VAL_SIZE = 500
TEST_SIZE = 1000

# Probe hyperparameters
PROBE_TRAINING_METHOD = "adam"  # "adam" or "sklearn"
PROBE_LR = 0.001
PROBE_WEIGHT_DECAY = 0.01
PROBE_NORMALIZE = True
PROBE_USE_BIAS = True
PROBE_SEED = 42
PROBE_EPOCHS = 100
PROBE_PATIENCE = 10
PROBE_C = 1.0  # For sklearn logistic regression (inverse of regularization strength)


# ============================================================================
# Step 1: Load Input Data
# ============================================================================

def load_input_data():
    """Load input data from HuggingFace dataset."""
    print("=" * 60)
    print("Step 1: Loading input data from HuggingFace")
    print("=" * 60)
    
    dataset = load_dataset(HF_DATASET_NAME, split=HF_DATASET_SPLIT)
    inputs = [item[HF_DATASET_COLUMN_INPUT] for item in dataset]
    
    print(f"Loaded {len(inputs)} input examples")
    return inputs


# ============================================================================
# Step 2: Generate Outputs
# ============================================================================

def load_model(model_name: str):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_outputs(inputs: List[str], model, tokenizer):
    """Generate outputs for inputs."""
    print("\n" + "=" * 60)
    print("Step 2: Generating outputs with model")
    print("=" * 60)
    
    all_outputs = []
    for i in range(0, len(inputs), BATCH_SIZE):
        batch_inputs = inputs[i:i + BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1}/{(len(inputs) + BATCH_SIZE - 1) // BATCH_SIZE}")
        
        # Tokenize
        encoded = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=1024
        ).to(model.device)
        
        # Generate
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=GENERATION_TEMPERATURE if GENERATION_TEMPERATURE > 0 else None,
                do_sample=(GENERATION_TEMPERATURE > 0),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        input_lengths = encoded['input_ids'].shape[1]
        generated_tokens = outputs[:, input_lengths:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        all_outputs.extend(decoded)
    
    print(f"Generated {len(all_outputs)} outputs")
    return all_outputs


# ============================================================================
# Step 3: Classify with GPT
# ============================================================================

async def classify_with_gpt(inputs: List[str], outputs: List[str]) -> List[Dict]:
    """Classify input-output pairs using GPT."""
    print("\n" + "=" * 60)
    print("Step 3: Classifying input-output pairs with GPT")
    print("=" * 60)
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(50)
    
    async def classify_single(input_text: str, output_text: str) -> Optional[Dict]:
        """Classify a single input-output pair."""
        async with semaphore:
            user_prompt = f"Input: {input_text}\n\nOutput: {output_text}"
            try:
                response = await client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                result = json.loads(response.choices[0].message.content)
                return {
                    'input': input_text,
                    'output': output_text,
                    'answer': result.get('answer', 5),
                    'confidence': result.get('confidence', 5),
                }
            except Exception as e:
                print(f"Error classifying example: {e}")
                return None
    
    print(f"Labeling {len(inputs)} examples with {GPT_MODEL}...")
    tasks = [classify_single(inp, out) for inp, out in zip(inputs, outputs)]
    gathered = await asyncio.gather(*tasks)
    results = [r for r in gathered if r is not None]
    
    # Classify labels and filter by confidence
    for item in results:
        answer = item['answer']
        if answer <= POSITIVE_THRESHOLD:
            item['label'] = 'positive'
        elif answer >= NEGATIVE_THRESHOLD:
            item['label'] = 'negative'
        else:
            item['label'] = 'ambiguous'
    
    # Filter by confidence threshold
    filtered_results = [r for r in results if r['confidence'] >= CONFIDENCE_THRESHOLD]
    
    # Count label distribution
    label_counts = Counter(item['label'] for item in filtered_results)
    
    print(f"Labeled {len(filtered_results)} examples (after confidence filtering)")
    print(f"Label distribution: {dict(label_counts)}")
    
    return filtered_results


# ============================================================================
# Step 4: Balance and Split Dataset
# ============================================================================

def balance_and_split_dataset(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Balance dataset and split into train, val, and test sets. Returns lists of dicts."""
    print("\n" + "=" * 60)
    print("Step 4: Balancing dataset and splitting into train-val-test")
    print("=" * 60)
    
    # Filter to only positive and negative, add label_binary
    filtered = []
    for item in data:
        if item['label'] in ['positive', 'negative']:
            item['label_binary'] = 1 if item['label'] == 'positive' else 0
            filtered.append(item)
    
    # Separate by label
    positive = [item for item in filtered if item['label_binary'] == 1]
    negative = [item for item in filtered if item['label_binary'] == 0]
    min_count = min(len(positive), len(negative))
    
    print(f"Balancing: {len(positive)} positive, {len(negative)} negative")
    print(f"Using {min_count} samples per class")
    
    # Sample balanced subsets
    np.random.seed(42)
    positive_indices = np.random.choice(len(positive), size=min_count, replace=False)
    negative_indices = np.random.choice(len(negative), size=min_count, replace=False)
    
    positive_balanced = [positive[i] for i in positive_indices]
    negative_balanced = [negative[i] for i in negative_indices]
    
    # Combine and shuffle
    balanced = positive_balanced + negative_balanced
    np.random.shuffle(balanced)
    
    # Split into train, val, test
    total = len(balanced)
    train_end = int(total * TRAIN_SIZE / (TRAIN_SIZE + VAL_SIZE + TEST_SIZE))
    val_end = train_end + int(total * VAL_SIZE / (TRAIN_SIZE + VAL_SIZE + TEST_SIZE))
    
    splits = {
        'train': balanced[:train_end],
        'val': balanced[train_end:val_end],
        'test': balanced[val_end:val_end + TEST_SIZE],
    }
    
    print("\nSplit sizes:")
    for split_name, split_data in splits.items():
        pos_count = sum(1 for item in split_data if item['label_binary'] == 1)
        print(f"  {split_name}: {len(split_data)} samples ({pos_count} positive)")
    
    return splits


# ============================================================================
# Step 5: Get Activations
# ============================================================================

def format_chat_prompt(tokenizer, input_text: str, output_text: str) -> str:
    """Format input-output pair as a chat prompt."""
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def get_activations_for_splits(splits: Dict[str, List[Dict]], model, tokenizer) -> Dict[str, Dict[str, torch.Tensor]]:
    """Get activations for all splits."""
    print("\n" + "=" * 60)
    print("Step 5: Getting activations for input-output pairs")
    print("=" * 60)
    
    # Get model layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError("Could not find model layers")
    
    all_activations = {}
    captured_activations = []
    
    def activation_hook(module, input, output):
        """Hook to capture activations."""
        if isinstance(output, tuple):
            captured_activations.append(output[0].detach())
        else:
            captured_activations.append(output.detach())
    
    for split_name, split_data in splits.items():
        print(f"\nProcessing {split_name} split ({len(split_data)} examples)...")
        
        split_activations = []
        split_masks = []
        for i in range(0, len(split_data), BATCH_SIZE):
            batch = split_data[i:i + BATCH_SIZE]
            print(f"  Batch {i // BATCH_SIZE + 1}/{(len(split_data) + BATCH_SIZE - 1) // BATCH_SIZE}")
            
            # Format and tokenize
            formatted_prompts = [
                format_chat_prompt(tokenizer, item['input'], item['output'])
                for item in batch
            ]
            
            encoded = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=2048
            ).to(model.device)
            
            # Register hook and forward pass
            captured_activations.clear()
            hook_handle = layers[LAYER].register_forward_hook(activation_hook)
            
            model.eval()
            with torch.no_grad():
                _ = model(**encoded)
            
            activations = captured_activations[0]  # [batch, seq_len, hidden_dim]
            hook_handle.remove()
            
            # Store full sequence activations and attention masks
            split_activations.append(activations.cpu())
            split_masks.append(encoded['attention_mask'].cpu())
        
        all_activations[split_name] = {
            'activations': torch.cat(split_activations, dim=0),  # [n_samples, seq_len, hidden_dim]
            'attention_mask': torch.cat(split_masks, dim=0),  # [n_samples, seq_len]
        }
        print(f"  {split_name} activations shape: {all_activations[split_name]['activations'].shape}")
    
    return all_activations


# ============================================================================
# Step 6: Create and Train Probe
# ============================================================================

def mean_pool_activations(activations: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool activations across sequence length, masking padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    masked_activations = activations * mask
    pooled = masked_activations.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled


# Option 1: PyTorch linear probe
# ============================================================================

class TorchLinearProbe(nn.Module):
    """PyTorch linear probe for binary classification with mean pooling aggregation."""
    
    def __init__(self, input_dim: int, normalize: bool = True, use_bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=use_bias)
        self.normalize = normalize
        self.mean = None
        self.std = None
    
    def fit_normalization(self, activations: torch.Tensor, attention_mask: torch.Tensor):
        """Fit normalization parameters on pooled activations."""
        if self.normalize:
            # Pool first, then compute stats
            pooled = mean_pool_activations(activations, attention_mask)
            self.mean = pooled.mean(dim=0, keepdim=True)
            self.std = pooled.std(dim=0, keepdim=True)
            self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
    
    def normalize_input(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize input."""
        if self.normalize and self.mean is not None:
            return (X - self.mean.to(X.device)) / self.std.to(X.device)
        return X
    
    def forward(self, activations: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass: pool, normalize, then linear layer."""
        # Pool activations: [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
        pooled = mean_pool_activations(activations, attention_mask)
        # Normalize
        pooled_norm = self.normalize_input(pooled)
        # Linear layer
        return self.linear(pooled_norm).squeeze(-1)


def create_and_train_probe_adam(
    train_activations: torch.Tensor,
    train_masks: torch.Tensor,
    train_labels: torch.Tensor,
    val_activations: Optional[torch.Tensor] = None,
    val_masks: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
) -> TorchLinearProbe:
    """Create and train a PyTorch linear probe using Adam optimizer."""
    print("\n" + "=" * 60)
    print("Step 6: Creating and training probe with Adam optimizer")
    print("=" * 60)
    
    # Create probe
    input_dim = train_activations.shape[2]  # hidden_dim
    probe = TorchLinearProbe(input_dim, normalize=PROBE_NORMALIZE, use_bias=PROBE_USE_BIAS)
    print(f"Created TorchLinearProbe with input_dim={input_dim}")
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probe = probe.to(device)
    train_activations = train_activations.to(device)
    train_masks = train_masks.to(device)
    train_labels = train_labels.float().to(device)
    
    # Fit normalization
    probe.fit_normalization(train_activations, train_masks)
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
    
    train_dataset = TensorDataset(train_activations, train_masks, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Setup validation if available
    val_loader = None
    if val_activations is not None and val_labels is not None:
        val_activations = val_activations.to(device)
        val_masks = val_masks.to(device)
        val_labels = val_labels.float().to(device)
        val_dataset = TensorDataset(val_activations, val_masks, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(PROBE_EPOCHS):
        # Train
        probe.train()
        train_loss = 0.0
        for acts_batch, masks_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = probe(acts_batch, masks_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        if val_loader is not None:
            probe.eval()
            val_loss = 0.0
            with torch.no_grad():
                for acts_batch, masks_batch, y_batch in val_loader:
                    logits = probe(acts_batch, masks_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = probe.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= PROBE_PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
    
    # Load best model
    if best_state is not None:
        probe.load_state_dict(best_state)
    
    print("Training complete!")
    return probe


# Option 2: scikit-learn logistic regression probe
# ============================================================================

def create_and_train_probe_sklearn(
    train_activations: torch.Tensor,
    train_masks: torch.Tensor,
    train_labels: torch.Tensor,
    val_activations: Optional[torch.Tensor] = None,
    val_masks: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
) -> LogisticRegression:
    """Create and train a scikit-learn logistic regression probe."""
    print("\n" + "=" * 60)
    print("Step 6: Creating and training probe with scikit-learn LogisticRegression")
    print("=" * 60)
    
    # Mean pool activations: [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
    print("Mean pooling activations...")
    train_pooled = mean_pool_activations(train_activations, train_masks)
    train_X = train_pooled.cpu().numpy()
    train_y = train_labels.cpu().numpy()
    
    if val_activations is not None:
        val_pooled = mean_pool_activations(val_activations, val_masks)
        val_X = val_pooled.cpu().numpy()
        val_y = val_labels.cpu().numpy()
    else:
        val_X = None
        val_y = None
    
    # Normalize if needed
    if PROBE_NORMALIZE:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        if val_X is not None:
            val_X = scaler.transform(val_X)
    else:
        scaler = None
    
    # Train logistic regression
    print("Fitting LogisticRegression...")
    clf = LogisticRegression(
        C=PROBE_C,
        fit_intercept=PROBE_USE_BIAS,
        max_iter=1000,
        random_state=PROBE_SEED,
        solver='lbfgs',
    )
    clf.fit(train_X, train_y)
    
    # Store scaler for later use
    clf.scaler = scaler
    
    # Print validation accuracy if available
    if val_X is not None:
        val_pred = clf.predict(val_X)
        val_acc = accuracy_score(val_y, val_pred)
        print(f"Validation accuracy: {val_acc:.4f}")
    
    print("Training complete!")
    return clf


# ============================================================================
# Step 7: Evaluate Probe
# ============================================================================

def evaluate_probe(
    probe,
    test_activations: torch.Tensor,
    test_masks: torch.Tensor,
    test_labels: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate the probe on test set."""
    print("\n" + "=" * 60)
    print("Step 7: Evaluating probe")
    print("=" * 60)
    
    # Determine probe type by checking if it's a sklearn LogisticRegression
    if isinstance(probe, LogisticRegression):
        # sklearn probe
        test_pooled = mean_pool_activations(test_activations, test_masks)
        test_X = test_pooled.cpu().numpy()
        
        if hasattr(probe, 'scaler') and probe.scaler is not None:
            test_X = probe.scaler.transform(test_X)
        
        probs = probe.predict_proba(test_X)[:, 1]
        preds = probe.predict(test_X)
    else:
        # PyTorch probe (or any other non-sklearn probe)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        probe.eval()
        
        with torch.no_grad():
            test_activations = test_activations.to(device)
            test_masks = test_masks.to(device)
            logits = probe(test_activations, test_masks)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
    
    y_true = test_labels.numpy()
    y_proba = probs
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, preds)
    auroc = roc_auc_score(y_true, y_proba)
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    target_fpr = 0.01
    idx = np.argmax(fpr >= target_fpr)
    tpr_at_1_fpr = tpr[idx] if idx < len(tpr) else 0.0
    
    results = {
        'accuracy': accuracy,
        'auroc': auroc,
        'tpr_at_1_fpr': tpr_at_1_fpr,
    }
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 60)
    
    return results


# ============================================================================
# Main Pipeline
# ============================================================================

async def main():
    """Run the complete pipeline."""
    print("\n" + "=" * 60)
    print("STANDALONE PROBE EXPERIMENT")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(PROBE_SEED)
    np.random.seed(PROBE_SEED)
    
    # Step 1: Load input data
    inputs = load_input_data()
    
    # Step 2: Generate outputs
    model, tokenizer = load_model(MODEL_NAME)
    outputs = generate_outputs(inputs, model, tokenizer)
    
    # Step 3: Classify with GPT
    labeled_data = await classify_with_gpt(inputs, outputs)
    
    # Step 4: Balance and split
    splits = balance_and_split_dataset(labeled_data)
    
    # Step 5: Get activations
    activations = get_activations_for_splits(splits, model, tokenizer)
    
    # Prepare labels
    labels = {
        'train': torch.tensor([item['label_binary'] for item in splits['train']], dtype=torch.float32),
        'val': torch.tensor([item['label_binary'] for item in splits['val']], dtype=torch.float32),
        'test': torch.tensor([item['label_binary'] for item in splits['test']], dtype=torch.float32),
    }
    
    # Step 6: Create and train probe
    if PROBE_TRAINING_METHOD == "adam":
        probe = create_and_train_probe_adam(
            activations['train']['activations'],
            activations['train']['attention_mask'],
            labels['train'],
            activations['val']['activations'],
            activations['val']['attention_mask'],
            labels['val'],
        )
    elif PROBE_TRAINING_METHOD == "sklearn":
        probe = create_and_train_probe_sklearn(
            activations['train']['activations'],
            activations['train']['attention_mask'],
            labels['train'],
            activations['val']['activations'],
            activations['val']['attention_mask'],
            labels['val'],
        )
    else:
        raise ValueError(f"Unknown training method: {PROBE_TRAINING_METHOD}")
    
    # Step 7: Evaluate probe
    results = evaluate_probe(
        probe,
        activations['test']['activations'],
        activations['test']['attention_mask'],
        labels['test'],
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
