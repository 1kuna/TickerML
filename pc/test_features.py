import unittest
import pandas as pd
import numpy as np
import subprocess
import os
import logging
from pc.features import compute_technical_indicators

# Configure logging for the test
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global scope for original function ---
compute_technical_indicators_original = None
original_function_code = None

try:
    # Attempt to get the original function code from the previous commit
    # Ensure we are in the root of the git repository for the command to work correctly
    # Assuming the script is run from a context where 'pc' is a subdirectory
    git_root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    features_path_in_repo = git_root / 'pc' / 'features.py'
    
    original_function_code_bytes = subprocess.check_output(
        ['git', 'show', f'HEAD~1:{features_path_in_repo}'],
        cwd=git_root
    )
    original_function_code = original_function_code_bytes.decode('utf-8')
    
    # Extract the specific function definition
    # This is a bit fragile; depends on the function definition not changing too much.
    # A more robust way might involve AST parsing, but this is simpler for now.
    
    # Create a temporary namespace to exec the original code
    temp_namespace = {}
    # Exec the whole original file content to get all dependencies (like 'ta')
    exec(original_function_code, temp_namespace)
    
    # Get the function from the temporary namespace
    if 'compute_technical_indicators' in temp_namespace:
        compute_technical_indicators_original = temp_namespace['compute_technical_indicators']
        logger.info("Successfully loaded original 'compute_technical_indicators' from git history.")
    else:
        logger.warning("Could not find 'compute_technical_indicators' in the original file content.")
        compute_technical_indicators_original = compute_technical_indicators # Fallback
        original_function_code = "ORIGINAL_NOT_FOUND_IN_FILE"


except subprocess.CalledProcessError as e:
    logger.warning(f"Failed to retrieve original function from git: {e}. Output: {e.output}. Stderr: {e.stderr}")
    logger.warning("Falling back to comparing the refactored function against itself.")
    compute_technical_indicators_original = compute_technical_indicators # Fallback
    original_function_code = "GIT_SHOW_FAILED"
except FileNotFoundError: # git command not found
    logger.warning("Git command not found. Falling back to comparing the refactored function against itself.")
    compute_technical_indicators_original = compute_technical_indicators # Fallback
    original_function_code = "GIT_NOT_FOUND"
except Exception as e:
    logger.error(f"An unexpected error occurred while trying to load original function: {e}")
    compute_technical_indicators_original = compute_technical_indicators # Fallback
    original_function_code = f"UNEXPECTED_ERROR: {e}"


class TestTechnicalIndicators(unittest.TestCase):

    def _create_sample_df(self, num_rows=100):
        """Creates a sample DataFrame for testing."""
        np.random.seed(0)  # Set seed for reproducibility
        data = {
            'timestamp': pd.to_datetime(np.arange(1672531200000, 1672531200000 + num_rows * 60000, 60000), unit='ms'),
            'open': np.random.rand(num_rows) * 100 + 1000,
            'high': np.random.rand(num_rows) * 50 + 1050, # Will be adjusted
            'low': np.random.rand(num_rows) * 50 + 950,   # Will be adjusted
            'close': np.random.rand(num_rows) * 100 + 1000,
            'volume': np.random.rand(num_rows) * 10 + 1
        }
        sample_df = pd.DataFrame(data)
        
        # Ensure 'high' is the max of o,h,l,c and 'low' is the min for consistency
        sample_df['high'] = sample_df[['open', 'high', 'low', 'close']].max(axis=1)
        sample_df['low'] = sample_df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Ensure 'open', 'close' are within 'low' and 'high'
        sample_df['open'] = np.clip(sample_df['open'], sample_df['low'], sample_df['high'])
        sample_df['close'] = np.clip(sample_df['close'], sample_df['low'], sample_df['high'])
        
        return sample_df

    def test_technical_indicators_consistency(self):
        """
        Tests if the refactored compute_technical_indicators function 
        produces the same output as the original version.
        """
        if compute_technical_indicators_original is None:
            self.skipTest("Original function could not be loaded. Skipping consistency test.")
            return

        if compute_technical_indicators_original == compute_technical_indicators:
            self.skipTest("Original function is identical to the refactored one. Skipping consistency test to avoid masking potential retrieval issues.")

        sample_df = self._create_sample_df(num_rows=100)
        
        # It's important to pass a copy to each function
        df_original = compute_technical_indicators_original(sample_df.copy())
        df_refactored = compute_technical_indicators(sample_df.copy())

        # Assert column names are identical and in the same order
        self.assertListEqual(list(df_original.columns), list(df_refactored.columns),
                             "Column names or order differ between original and refactored DataFrames.")

        # Assert indices are identical
        pd.testing.assert_index_equal(df_original.index, df_refactored.index,
                                      msg="Indices differ between original and refactored DataFrames.")
        
        # Detailed column-by-column comparison for better error messages
        for col in df_original.columns:
            try:
                pd.testing.assert_series_equal(
                    df_original[col], 
                    df_refactored[col], 
                    check_dtype=False, 
                    atol=1e-9, # Adjusted tolerance for floating point comparisons
                    obj=f"DataFrame column '{col}'"
                )
            except AssertionError as e:
                # If a specific column fails, it might be due to the volume_sma fix.
                # Log this and provide more context.
                if col == 'volume_sma' and "ORIGINAL_NOT_FOUND_IN_FILE" not in (original_function_code or "") and "GIT_SHOW_FAILED" not in (original_function_code or ""):
                    logger.warning(
                        f"AssertionError for column '{col}': {e}. "
                        "This might be expected due to the 'volume_sma' bug fix in the refactored version. "
                        "The original version had series_close and series_volume parameters swapped for this indicator."
                    )
                    # To make the test pass despite this known difference, we could:
                    # 1. Skip comparing this column if the original was successfully loaded.
                    # 2. Or, re-calculate this specific column for df_original using the corrected logic.
                    # For now, we'll let it fail to highlight the difference.
                    # If this is the *only* difference, the test has served its purpose.
                
                # Re-raise the assertion error to mark the test as failed or error
                raise e


        # Fallback to overall frame equal if all columns passed (or if we want a final check)
        # This might be redundant if column-by-column check is comprehensive
        try:
            pd.testing.assert_frame_equal(
                df_original, 
                df_refactored, 
                check_dtype=False, 
                atol=1e-9 # Adjusted tolerance
            )
        except AssertionError as e:
            logger.error(f"Overall pd.testing.assert_frame_equal failed: {e}")
            # If original_function_code is available, log snippets for debugging
            if original_function_code and len(original_function_code) < 2000: # Avoid huge logs
                 logger.info(f"Original function code snippet:\n{original_function_code[:1000]}...")
            raise

if __name__ == '__main__':
    # This allows running the test directly using `python pc/test_features.py`
    unittest.main()
