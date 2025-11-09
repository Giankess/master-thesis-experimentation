"""FinBERT-based text processing for financial text."""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Union, Optional
from tqdm import tqdm


class FinBERTProcessor:
    """Process financial text using FinBERT for embeddings and tokenization."""

    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone",
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        """
        Initialize FinBERT processor.

        Args:
            model_name: HuggingFace model name for FinBERT
            max_length: Maximum sequence length
            device: Device to use (cuda/cpu). Auto-detected if None
        """
        self.model_name = model_name
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading FinBERT model: {model_name}")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def tokenize(
        self, texts: Union[str, List[str]], return_tensors: bool = True
    ) -> dict:
        """
        Tokenize text(s).

        Args:
            texts: Single text or list of texts
            return_tensors: Whether to return PyTorch tensors

        Returns:
            Dictionary with tokenized inputs
        """
        if isinstance(texts, str):
            texts = [texts]

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt" if return_tensors else None,
        )

        if return_tensors:
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

        return encoding

    def get_embeddings(
        self, texts: Union[str, List[str]], batch_size: int = 8, pooling: str = "cls"
    ) -> np.ndarray:
        """
        Generate embeddings for text(s) using FinBERT.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            pooling: Pooling strategy ('cls', 'mean', 'max')

        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
                batch_texts = texts[i : i + batch_size]
                inputs = self.tokenize(batch_texts)

                outputs = self.model(**inputs)

                # Apply pooling strategy
                if pooling == "cls":
                    # Use [CLS] token embedding
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif pooling == "mean":
                    # Mean pooling over all tokens
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = (
                        attention_mask.unsqueeze(-1)
                        .expand(token_embeddings.size())
                        .float()
                    )
                    embeddings = (
                        torch.sum(token_embeddings * input_mask_expanded, 1)
                        / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    ).cpu().numpy()
                elif pooling == "max":
                    # Max pooling over all tokens
                    embeddings = torch.max(outputs.last_hidden_state, dim=1)[0].cpu().numpy()
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling}")

                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def process_texts(
        self,
        texts: List[str],
        batch_size: int = 8,
        pooling: str = "cls",
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Process multiple texts and return embeddings.

        Args:
            texts: List of texts to process
            batch_size: Batch size for processing
            pooling: Pooling strategy
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings
        """
        return self.get_embeddings(texts, batch_size=batch_size, pooling=pooling)

    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings.

        Returns:
            Embedding dimension size
        """
        # Get hidden size from model config
        return self.model.config.hidden_size

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before tokenization.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        # Basic preprocessing
        text = text.strip()
        # Remove multiple spaces
        text = " ".join(text.split())
        return text

    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """
        Preprocess multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]
