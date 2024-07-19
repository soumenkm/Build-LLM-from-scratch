import torch, tqdm, tiktoken, sys, random
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent)) # Adds instruct_fine_tuning in the search list
sys.path.append(str(Path(__file__).parent.parent)) # Adds build-llm in the search list

from gpt_model.gpt import GPTModel
from gpt_model.load_openai_gpt2 import get_openai_gpt2_parameters
from gpt_model.gpt_download import download_and_load_gpt2
from typing import Tuple, List