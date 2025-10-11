"""
Base class for prompt datasets with unified interface.

This class standardizes how all "_behaviour.py" scripts work, handling data downloading,
train/test separation, and providing a consistent interface across all dataset types.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from probe_gen.annotation.interface_dataset import Dataset


class PromptDataset(ABC):
    """
    Base class for prompt datasets with unified interface.
    
    Each dataset subclass must implement:
    - `_create_dataset(mode, n_samples, skip)`: Generate the actual dataset
    - Optional: `download_data()` if external data needs to be downloaded
    
    By default, `generate_data()` creates both train and test datasets with automatic
    separation. Users can override the skip values or use single-dataset mode.
    
    This class handles:
    - Automatic data downloading
    - Train/test separation with configurable logic
    - Consistent save/load interface
    - Input validation and error handling
    """
    
    def __init__(self, dataset_name: str, default_train_test_gap: int = 100):
        """
        Initialize the prompt dataset.
        
        Args:
            dataset_name: Name identifier for this dataset type
            default_train_test_gap: Default gap between train and test data
        """
        self.dataset_name = dataset_name
        self.default_train_test_gap = default_train_test_gap
        self._data_downloaded = False
    
    @abstractmethod
    def _create_dataset(self, mode: str, n_samples: int, skip: int) -> Dataset:
        """
        Create a dataset with the specified parameters.
        
        This is the core method that subclasses must implement. It should
        generate a Dataset object with the requested number of samples.
        
        Args:
            mode: "train" or "test" 
            n_samples: Number of samples to generate
            skip: Number of samples to skip from the beginning
            
        Returns:
            Dataset object with the generated samples
            
        Raises:
            ValueError: If not enough samples are available
        """
        pass
    
    def download_data(self) -> None:
        """
        Download any required data files (CSV, datasets from HuggingFace, etc.).
        
        This method should be overridden by subclasses that need to download
        external data. The base implementation does nothing.
        """
        pass
    
    def generate_data(
        self,
        mode: str = None,
        output_file: str = None,
        n_samples: int = None,
        skip: int = 0
    ) -> Dict[str, Dataset]:
        """
        Generate train and/or test datasets.
        
        This method supports three modes:
        1. Both train and test: Specify both train_samples and test_samples
        2. Train only: Specify only train_samples  
        3. Test only: Specify only test_samples
        
        When creating both datasets, test data automatically starts after train data 
        plus the default gap to ensure no overlap between train and test sets.
        
        Args:
            save_dir: Directory to save the datasets
            train_samples: Number of training samples (optional if only creating test set)
            test_samples: Number of test samples (optional if only creating train set)
            skip: Skip value for single dataset mode
            save_location: Save path for single dataset mode


            # Alternative single-dataset mode:
            only specify one of train_samples or test_samples

            
        Returns:
            Dictionary with requested datasets ('train' and/or 'test' keys)
            
        Raises:
            ValueError: If unable to generate enough samples or invalid parameter combinations
        """
        self._ensure_data_downloaded()

        if mode != "train" and mode != "test":
            raise ValueError("Mode must be 'train' or 'test'")
        if n_samples < 0: 
            raise ValueError("n_samples must be positive")
        if output_file is None:
            raise ValueError("output_file must be provided")

        # Ensure save directory exists
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file = str(output_file)

        if mode == "test":
            skip += self.default_train_test_gap

        # Check if using single dataset mode (backwards compatibility)
        self._validate_inputs(mode, n_samples, skip)
        print(f"Generating {n_samples} {mode} samples (skip={skip})")
        dataset = self._create_dataset(mode, n_samples, skip)
        
        found_enough_samples = len(dataset) >= n_samples
        
        self._save_dataset(dataset, save_location=output_file)
        print(f"✓ Successfully generated {len(dataset)} {mode} samples, (skipped {skip} samples), saved to {output_file}")

        return found_enough_samples # {mode: dataset}    
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about this dataset.
        
        Returns:
            Dictionary with dataset metadata
        """
        return {
            'name': self.dataset_name,
            'class': self.__class__.__name__,
            'description': self.__doc__ or "No description available",
            'train_test_gap': self.default_train_test_gap
        }
    
    def _validate_inputs(self, mode: str, n_samples: int, skip: int) -> None:
        """Validate input parameters."""
        if mode not in ["train", "test"]:
            raise ValueError(f"Mode must be 'train' or 'test', got '{mode}'")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if skip < 0:
            raise ValueError(f"skip must be non-negative, got {skip}")
    
    def _ensure_data_downloaded(self) -> None:
        """Ensure data is downloaded (only call once)."""
        if not self._data_downloaded:
            print(f"Downloading data for {self.dataset_name}...")
            self.download_data()
            self._data_downloaded = True

    def _save_dataset(self, dataset: Dataset, save_location:str) -> None:
        """Save dataset to JSON file."""
        # Convert dataset to pandas and then to JSON
        df = dataset.to_pandas()

        # Save as JSON lines format for easy loading
        with open(save_location, 'w') as f:
            for _, row in df.iterrows():
                json.dump(row.to_dict(), f)
                f.write('\n')
        
        print(f"Dataset saved to {save_location}")

    # OLD
    # def generate_data(
    #     self,
    #     #mode: str = "train_test",
    #     skip: int = 0,
    #     save_dir: str = None,
    #     train_samples: Optional[int] = None,
    #     test_samples: Optional[int] = None,
    #     #n_samples: Optional[int] = None,
    # ) -> Dict[str, Dataset]:
    #     """
    #     Generate train and/or test datasets.
        
    #     This method supports three modes:
    #     1. Both train and test: Specify both train_samples and test_samples
    #     2. Train only: Specify only train_samples  
    #     3. Test only: Specify only test_samples
        
    #     When creating both datasets, test data automatically starts after train data 
    #     plus the default gap to ensure no overlap between train and test sets.
        
    #     Args:
    #         save_dir: Directory to save the datasets
    #         train_samples: Number of training samples (optional if only creating test set)
    #         test_samples: Number of test samples (optional if only creating train set)
    #         skip: Skip value for single dataset mode
    #         save_location: Save path for single dataset mode


    #         # Alternative single-dataset mode:
    #         only specify one of train_samples or test_samples

            
    #     Returns:
    #         Dictionary with requested datasets ('train' and/or 'test' keys)
            
    #     Raises:
    #         ValueError: If unable to generate enough samples or invalid parameter combinations
    #     """
    #     self._ensure_data_downloaded()


    #     mode = None
    #     if train_samples is None: 
    #         mode = "test"
    #         n_samples = test_samples
    #     if test_samples is None: 
    #         mode = "train"
    #         n_samples = train_samples


    #     # Ensure save directory exists
    #     if save_dir is not None:    
    #         save_dir = Path(save_dir)
    #         save_dir.mkdir(parents=True, exist_ok=True)
    #         if mode is not None:
    #             save_path = str(save_dir / f"{self.dataset_name}_{mode}.json")
    #         else:
    #             train_save_path = str(save_dir / f"{self.dataset_name}_train.json")
    #             test_save_path = str(save_dir / f"{self.dataset_name}_test.json")
    #     else: 
    #         default_save_location = self._get_default_save_location()

    #         if mode is not None:
    #             save_path = default_save_location / f"{self.dataset_name}_{mode}.json"
    #         else:
    #             train_save_path = default_save_location / f"{self.dataset_name}_train.json"
    #             test_save_path = default_save_location / f"{self.dataset_name}_test.json"

        
    #     # Check if using single dataset mode (backwards compatibility)
    #     if mode:
    #         self._validate_inputs(mode, n_samples, skip)
    #         print(f"Generating {n_samples} {mode} samples (skip={skip})")
    #         dataset = self._create_dataset(mode, n_samples, skip)
            
    #         if len(dataset) < n_samples:
    #             raise ValueError(f"Generated only {len(dataset)} samples, needed {n_samples}")
            
    #         self._save_dataset(dataset, save_location=save_path)
    #         print(f"✓ Successfully generated {len(dataset)} {mode} samples, (skipped {skip} samples), saved to {save_path}")

    #         return {mode: dataset}
        
    #     # Validate that at least one split is requested
    #     if train_samples is None and test_samples is None:
    #         raise ValueError("Must specify at least one of train_samples or test_samples")
        
    #     # Validate positive sample counts
    #     if train_samples is not None and train_samples <= 0:
    #         raise ValueError("train_samples must be positive")
    #     if test_samples is not None and test_samples <= 0:
    #         raise ValueError("test_samples must be positive")
        
    #     # Set default skip values
    #     if skip is None:
    #         skip = 0
        
    #     results = {}
        
    #     # Generate training data if requested
    #     if train_samples is not None:
    #         print(f"Generating {train_samples} train samples (skip={skip})")
    #         train_dataset = self._create_dataset("train", train_samples, skip)
    #         if len(train_dataset) < train_samples:
    #             raise ValueError(f"Generated only {len(train_dataset)} train samples, needed {train_samples}")
            
    #         self._save_dataset(train_dataset, save_location=train_save_path)
    #         results['train'] = train_dataset
    #         print(f"✓ Successfully generated {len(train_dataset)} train samples")
        
    #     # Generate test data if requested
    #     if test_samples is not None:
    #         # Calculate test skip if not provided and train data exists
    #         test_skip = skip + train_samples + self.default_train_test_gap
    #         print(f"Generating {test_samples} test samples (skip={test_skip})")
            
    #         test_dataset = self._create_dataset("test", test_samples, test_skip)
    #         if len(test_dataset) < test_samples:
    #             raise ValueError(f"Generated only {len(test_dataset)} test samples, needed {test_samples}")
        
    #         self._save_dataset(test_dataset, test_save_path)
    #         results['test'] = test_dataset
    #         print(f"✓ Successfully generated {len(test_dataset)} test samples")
        
    #     # Print summary
    #     if len(results) > 1:
    #         print("\n✓ Successfully created balanced dataset:")
    #         if 'train' in results:
    #             print(f"  Train: {len(results['train'])} samples (saved to {train_save_path})")
    #         if 'test' in results:
    #             print(f"  Test: {len(results['test'])} samples (saved to {test_save_path})")
    #     else:
    #         split_name = list(results.keys())[0]
    #         print(f"\n✓ Successfully created {split_name} dataset with {len(results[split_name])} samples")
        
    #     return results
