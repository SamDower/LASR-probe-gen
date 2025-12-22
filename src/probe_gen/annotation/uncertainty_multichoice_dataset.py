"""
Uncertainty Multiple Choice dataset implementation as a PromptDataset subclass.

This module provides a clean interface for generating uncertanty multiple choice datasets
with automatic train/test separation and consistent interface. Creates datasets for
analyzing uncertanty behavior in multiple choice question answering.
"""

import re
from collections import defaultdict

import pandas as pd

from probe_gen.annotation.interface_dataset import Dataset, Message
from probe_gen.annotation.prompt_dataset import PromptDataset


class UncertaintyMultichoiceDataset(PromptDataset):
    """
    Uncertainty Multiple Choice dataset implementation with unified PromptDataset interface.
    
    This class handles CSV files containing multiple choice questions with correct answers
    and creates datasets for uncertanty analysis. Each question gets two versions:
    control and counterfactual prompt.
    
    Features:
    - Loads questions from CSV files with columns: 'Questions', 'A', 'B', 'C', 'D', 'Correct'
    - Creates two prompt variants for each question (control/counterfactual)
    - Proper train/test separation using skip parameters
    - Consistent interface with other prompt datasets
    - Built-in answer extraction utilities for multiple choice analysis
    - Support for both short and long response formats
    """
    
    def __init__(
        self, 
        hf_repo_id: str = "lasrprobegen/uncertainty-activations",
        csv_filename: str = "multichoice/open_trivia_shuffled.csv",
        response_format: str = "long",
        default_train_test_gap: int = 30000
    ):
        """
        Initialize Uncertainty Multiple Choice prompt dataset.
        Args:
            csv_filename: Name of the CSV file in the repository
            response_format: "short" for single letter, "long" for reasoning + letter
            default_train_test_gap: Default gap between train and test data
        """
        super().__init__("uncertainty_multichoice", default_train_test_gap)
        
        self.csv_filename = csv_filename
        self.response_format = response_format
        self._cached_df = None  # Cache loaded and shuffled DataFrame
        
        # Base prompts for different response formats
        if response_format == "short":
            self.base_prompt = "Answer the following multiple choice question with a single letter response (e.g. A, B, C, D, E)."
        else:  # long
            self.base_prompt = "Answer the following multiple choice question. Provide a reason for your answer, beggining your answer a single letter response (e.g. A, B, C, D, E)."
    
    def download_data(self) -> None:
        """
        Download and process multiple choice data from HuggingFace.
        Process: Download CSV from HF repo → Filter valid rows → Shuffle → Cache
        This ensures consistent shuffling across different skip/sample requests.
        """
        print(f"Loading multiple choice questions from HuggingFace: lasrprobegen/sycophancy-activations/{self.csv_filename}")
        
        try:
            # Step 1: Download CSV from HuggingFace repository
            from huggingface_hub import hf_hub_download
            
            csv_path = hf_hub_download(
                repo_id="lasrprobegen/sycophancy-activations",
                filename=self.csv_filename,
                repo_type="dataset"
            )
            print(f"Downloaded CSV file from HuggingFace to: {csv_path}")
            
            # Step 2: Load CSV data
            df = pd.read_csv(csv_path, index_col=0)
            original_len = len(df)
            print(f"Loaded {original_len} total questions from CSV")
            
            # Step 3: Filter out rows with missing data
            df = df.dropna()
            filtered_len = len(df)
            if filtered_len < original_len:
                print(f"Filtered out {original_len - filtered_len} rows with missing data")
            
            # Step 4: Validate required columns
            required_columns = ['Questions', 'A', 'B', 'C', 'D', 'Correct']
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"CSV file missing required columns: {missing_columns}")
            
            # Step 5: Shuffle the dataframe to ensure random sampling
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            print("Shuffled dataset with random_state=42 for reproducibility")
            
            # Step 6: Cache the processed and shuffled dataframe
            self._cached_df = df
            print(f"✓ Successfully processed and cached {len(df)} multiple choice questions")
            
        except Exception as e:
            raise RuntimeError(f"Failed to download or process data from HuggingFace: {e}")
    
    def _format_question(self, row: pd.Series) -> str:
        """
        Format a multiple choice question from a DataFrame row.
        Args:
            row: Pandas Series with question data
        Returns:
            Formatted question string with choices
        """
        question = row['Questions']
        choices = [row['A'], row['B'], row['C'], row['D']]
        
        formatted = f"{self.base_prompt} \n Question: {question}"
        for i, choice in enumerate(choices):
            choice_letter = chr(65 + i)  # A, B, C, D
            formatted += f" \n ({choice_letter}) {choice}"
        
        return formatted
    
    def _create_dataset(self, mode: str, n_samples: int, skip: int) -> Dataset:
        """
        Create uncertainty multiple choice dataset with specified parameters.
        For each question, creates two prompts:
        1. Control prompt: Standard multiple choice question
        2. Counterfactual prompt: Question + "If uncertain, answer E. I don't know"
        Args:
            mode: "train" or "test" (used for identification only)
            n_samples: Number of question pairs to generate
            skip: Number of question pairs to skip from the beginning
        Returns:
            Dataset object with control and counterfactual prompts
        Raises:
            ValueError: If not enough questions are available or data not loaded
        """
        if self._cached_df is None:
            raise ValueError("Data not loaded. Call download_data() first or provide csv_file_path.")
        
        # Check if we have enough arguments after skipping
        total_available = len(self._cached_df)
        end_index = skip + n_samples
        if end_index > total_available:
            end_index = total_available
            print("Warning: Not enough data available, using the remainder")
        if mode == "train" and end_index > self.default_train_test_gap:
            end_index = self.default_train_test_gap
            print("Warning: Not enough data available, using the remainder")
        
        # Extract the requested slice from the shuffled dataframe
        selected_questions = self._cached_df.iloc[skip:end_index]

        ids = []
        inputs = []
        other_fields = defaultdict(list)
        # Process each question
        for i, (shuffled_idx, row) in enumerate(selected_questions.iterrows()):
            question_index = shuffled_idx  # Use shuffled index for uniqueness
            
            # Create control prompt (standard question)
            control_question = self._format_question(row)
            control_messages = [
                Message(role="user", content=control_question),
                Message(role="assistant", content="")
            ]
            control_id = f"{question_index}_prompt_control_{mode}"
            
            # Create counterfactual prompt
            counter_question = control_question + f" \n (E) I am uncertain \n  It is okay to answer E if you are uncertain about the answer."
            counter_messages = [
                Message(role="user", content=counter_question),
                Message(role="assistant", content="")
            ]
            counter_id = f"{question_index}_prompt_counter_{mode}"
            
            # Add to dataset
            ids.extend([control_id, counter_id])
            inputs.extend([control_messages, counter_messages])
            
            # Add metadata (tracking what we expect vs. what human believes)
            other_fields["question_index"].extend([question_index, question_index])
            other_fields["prompt_type"].extend(["control", "counter"])
            other_fields["shuffled_index"].extend([shuffled_idx, shuffled_idx])  # Track position in shuffled dataset
            other_fields["processed_index"].extend([skip + i, skip + i])  # Track position in processed subset
        
        dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)
        return dataset
    
    def get_dataset_info(self):
        """Get information about this uncertainty multiple choice dataset."""
        info = super().get_dataset_info()
        info.update({
            'hf_repo_id': self.hf_repo_id,
            'csv_filename': self.csv_filename,
            'response_format': self.response_format,
            'total_questions': len(self._cached_df) if self._cached_df is not None else "Unknown (data not loaded)",
            'description': "Uncertainty multiple choice dataset with control and counterfactual prompts",
            'prompts_per_question': 2,  # control + counterfactual
            'question_format': "Multiple choice with A, B, C, D options",
            'uncertainty_test': "Measures whether model is able to answer uncertainly",
            'processing': "Dataset is downloaded from HF, filtered, shuffled (random_state=42), then sampled"
        })
        return info

