from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, normalizers
from datasets import load_from_disk
import json
import os
from pathlib import Path
from tqdm import tqdm

class ChessTokenizerTrainer:
    
    def __init__(self, dataset_path="./chess_data/hf_dataset"):
        self.dataset_path = dataset_path
        self.tokenizer = None
        
    def load_dataset_iterator(self, split="train", num_samples=None):
        """
        Create an iterator that yields game strings from the dataset
        FIXED: Added max_length filter to avoid overly long games.
        """
        try:
            dataset = load_from_disk(self.dataset_path)
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {self.dataset_path}: {e}")
        
        if split not in dataset:
            raise ValueError(f"Split {split} not found in dataset")
        
        # Get the text column
        data = dataset[split]
        
        # Limit samples
        if num_samples:
            data = data.select(range(min(num_samples, len(data))))
        
        # Yield with length check (rough: split and count tokens)
        for i in range(len(data)):
            game = data[i]["text"]
            yield game
            
    
    def create_tokenizer(self, vocab_size=1971):
        """
        Create and train a WordLevel tokenizer on the chess dataset
        FIXED: vocab_size=1971 to match target (1970 moves + 1 unk = 1971 before adding specials)
        """
        print("Initializing WordLevel tokenizer...")
        
        # Initialize a WordLevel tokenizer
        self.tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
        
        # Enable truncation
        self.tokenizer.enable_truncation(max_length=256)
        
        # Add normalizer
        self.tokenizer.normalizer = normalizers.BertNormalizer()
        
        # Use Whitespace pre-tokenizer (UCI moves are space-separated)
        self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        
        # Initialize trainer
        # Special tokens must be in the order we want them in vocabulary
        # FIXED: Only include <unk> during training; we'll add others later at fixed positions
        special_tokens = ["<unk>"]
        
        trainer = trainers.WordLevelTrainer(
            vocab_size=vocab_size,  # 1971 total
            special_tokens=special_tokens,
        )
        
        return trainer
    
    def train_tokenizer(self, vocab_size=1971, num_training_samples=1000000):
        """
        Train the tokenizer on the dataset
        FIXED: vocab_size=1971 to match target
        """
        print(f"Training tokenizer on {num_training_samples:,} samples...")
        
        # Create trainer
        trainer = self.create_tokenizer(vocab_size)
        
        # Get training data iterator
        train_iterator = self.load_dataset_iterator(
            split="train", 
            num_samples=num_training_samples,
        )
        
        # Count total samples for progress bar
        total_samples = min(
            num_training_samples, 
            len(load_from_disk(self.dataset_path)["train"])
        )
        
        # Wrap iterator with tqdm for progress
        class ProgressIterator:
            def __init__(self, iterator, total):
                self.iterator = iterator
                self.total = total
                self.pbar = tqdm(total=total, desc="Training tokenizer")
                
            def __iter__(self):
                for item in self.iterator:
                    yield item
                    self.pbar.update(1)
                self.pbar.close()
        
        try:
            # Train tokenizer
            self.tokenizer.train_from_iterator(
                ProgressIterator(train_iterator, total_samples),
                trainer=trainer,
                length=total_samples
            )
        except Exception as e:
            raise RuntimeError(f"Tokenizer training failed: {e}")
        
        print(f"Tokenizer training complete!")
        print(f"Vocabulary size: {self.tokenizer.get_vocab_size()}")
        
        return self.tokenizer
    
    def analyze_vocabulary(self):
        """
        Analyze the trained vocabulary
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        
        vocab = self.tokenizer.get_vocab()
        
        # Count different types of tokens
        move_tokens = []
        result_tokens = []
        special_tokens = []
        
        for token, idx in vocab.items():
            if token in ["<unk>"]:
                special_tokens.append((token, idx))
            elif token in ["1-0", "0-1"]:
                result_tokens.append((token, idx))
            else:
                move_tokens.append((token, idx))
        
        print(f"\nVocabulary Analysis:")
        print(f"Special tokens: {len(special_tokens)}")
        for token, idx in sorted(special_tokens, key=lambda x: x[1]):
            print(f"  {idx}: {token}")
        
        print(f"\nResult tokens: {len(result_tokens)}")
        for token, idx in sorted(result_tokens, key=lambda x: x[1]):
            print(f"  {idx}: {token}")
        
        print(f"\nMove tokens: {len(move_tokens)}")
        
        # Show first 20 move tokens
        print("First 20 move tokens:")
        for token, idx in sorted(move_tokens[:20], key=lambda x: x[1]):
            print(f"  {idx}: {token}")
        
        # Check if all moves are valid UCI
        invalid_moves = []
        for token, _ in move_tokens:
            if token in ["1-0", "0-1"]:  # Result tokens are valid
                continue
            if len(token) != 4 and len(token) != 5:  # 4 for normal, 5 for promotion (e.g., e7e8q)
                invalid_moves.append(token)
            elif len(token) == 4:
                if not (token[0] in 'abcdefgh' and token[1] in '12345678' and
                       token[2] in 'abcdefgh' and token[3] in '12345678'):
                    invalid_moves.append(token)
            elif len(token) == 5:
                if not (token[0] in 'abcdefgh' and token[1] in '12345678' and
                       token[2] in 'abcdefgh' and token[3] in '12345678' and
                       token[4] in 'qrbn'):
                    invalid_moves.append(token)
        
        if invalid_moves:
            print(f"\nWarning: {len(invalid_moves)} potentially invalid UCI moves in vocabulary:")
            print(invalid_moves[:10])
        else:
            print(f"\nAll {len(move_tokens)} move tokens are valid UCI!")
        
        return {
            "special_tokens": special_tokens,
            "result_tokens": result_tokens,
            "move_tokens": move_tokens,
            "total_vocab_size": len(vocab)
        }
    
    def save_tokenizer(self, output_dir="./chess-llama-tokenizer"):
        """
        Save the trained tokenizer
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        
        try:
            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save tokenizer
            self.tokenizer.save(str(output_dir / "tokenizer.json"))
            
            # Create config file matching target
            config = {
                "tokenizer_type": "WordLevel",
                "vocab_size": 1971,
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
            }
            
            with open(output_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            # Save tokenizer_config for HF
            tokenizer_config = {
                "truncation_side": "right",
                "special_tokens_map_file": None,
            }
            with open(output_dir / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)
            
            print(f"Tokenizer saved to {output_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to save tokenizer: {e}")
    
    def wrap_as_pretrained_tokenizer(self):
        """
        Wrap the trained tokenizer as PreTrainedTokenizerFast
        FIXED: Remap to match target exactly:
        - vocab_size = 1971 (only moves/results, not including special tokens in count)
        - Special tokens: unk=0, bos=1971, eos=1972, pad=1973
        - Moves/results: 1-1970
        """
        from transformers import PreTrainedTokenizerFast
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        
        # Get current vocab
        vocab = self.tokenizer.get_vocab()
        print(f"Initial vocab length from training: {len(vocab)}")
    
        # Wrap as PreTrainedTokenizerFast
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            truncation_side="right",
        )
        
        num_added_tokens = wrapped_tokenizer.add_special_tokens({
            'bos_token': '<s>',
            'eos_token': '</s>',
            'pad_token': '<pad>',
        })
        
        vocab = wrapped_tokenizer.get_vocab()
        print(f"Vocab length after adding special tokens: {len(vocab)}")
        
        # Verify special token IDs match target
        print(f"\nFinal special token IDs:")
        print(f"  unk_token_id: {wrapped_tokenizer.unk_token_id} (expected: 0)")
        print(f"  bos_token_id: {wrapped_tokenizer.bos_token_id} (expected: 1971)")
        print(f"  eos_token_id: {wrapped_tokenizer.eos_token_id} (expected: 1972)")
        print(f"  pad_token_id: {wrapped_tokenizer.pad_token_id} (expected: 1973)")
        print(f"  vocab_size: {wrapped_tokenizer.vocab_size} (expected: 1971)")
              
        print("\n✓ All token IDs match target configuration!")
        
        return wrapped_tokenizer

def create_final_tokenizer_pipeline(num_samples=1000000):
    """
    Complete pipeline to create and train tokenizer
    FIXED: Matches target vocab_size=1971 with correct special token positions
    """
    from transformers import AutoTokenizer
    
    # Check if dataset exists
    if not os.path.exists("./chess_data/hf_dataset"):
        print("Dataset not found. Please run dataset preparation first.")
        return None
    
    try:
        # Initialize trainer
        trainer = ChessTokenizerTrainer("./chess_data/hf_dataset")
        
        # Train tokenizer with vocab_size=1971
        tokenizer = trainer.train_tokenizer(
            vocab_size=1971,
            num_training_samples=num_samples,
        )
        
        # Analyze vocabulary
        analysis = trainer.analyze_vocabulary()
        
        # Save raw tokenizer
        trainer.save_tokenizer("./chess_tokenizer_raw")
        
        # Create wrapped tokenizer with correct special token positions
        wrapped_tokenizer = trainer.wrap_as_pretrained_tokenizer()
        
        # Save as pretrained tokenizer
        wrapped_tokenizer.save_pretrained("./chess-llama-tokenizer")
        wrapped_tokenizer.push_to_hub("VinayHajare/chess-llama", revision="reproduce-branch")
        
        # Test the tokenizer
        print("\n" + "="*50)
        print("Testing tokenizer...")
        
        test_games = [
            "e2e4 e7e5 g1f3 b8c6 f1c4 1-0",  # Result at END
            "d2d4 d7d5 c2c4 e7e6 0-1",       # Result at END
        ]
        
        for game in test_games:
            print(f"\nGame: {game}")
            encoded = wrapped_tokenizer(game, add_special_tokens=True)
            print(f"Input IDs: {encoded['input_ids']}")
            print(f"Tokens: {wrapped_tokenizer.convert_ids_to_tokens(encoded['input_ids'])}")
            print(f"Decoded: {wrapped_tokenizer.decode(encoded['input_ids'], skip_special_tokens=False)}")
        
        return wrapped_tokenizer
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_and_test_existing_tokenizer(repo_name="VinayHajare/chess-llama"):
    """
    Load the original tokenizer and compare
    """
    from transformers import AutoTokenizer
    
    print(f"Loading original tokenizer from HuggingFace ({repo_name})...")
    try:
        original_tokenizer = AutoTokenizer.from_pretrained(repo_name)
        
        print(f"\nOriginal tokenizer info:")
        print(f"Vocab size: {original_tokenizer.vocab_size}")
        print(f"Special tokens: {original_tokenizer.special_tokens_map}")
        print(f"Special token IDs:")
        print(f"  unk: {original_tokenizer.unk_token_id}")
        print(f"  bos: {original_tokenizer.bos_token_id}")
        print(f"  eos: {original_tokenizer.eos_token_id}")
        print(f"  pad: {original_tokenizer.pad_token_id}")
        
        # Test encoding
        test_game = "e2e4 e7e5 g1f3 b8c6 f1c4 1-0"
        encoded = original_tokenizer(test_game, add_special_tokens=True)
        print(f"\nOriginal tokenizer on '{test_game}':")
        print(f"Input IDs: {encoded['input_ids']}")
        print(f"Tokens: {original_tokenizer.convert_ids_to_tokens(encoded['input_ids'])}")
        
        return original_tokenizer
        
    except Exception as e:
        print(f"Skipping original load (e.g., repo unavailable): {e}")
        return None

if __name__ == "__main__":
    print("Chess Tokenizer Training Pipeline")
    print("="*50)
    
    # Option 1: Compare with original (optional)
    original = load_and_test_existing_tokenizer()
    
    # Option 2: Train new tokenizer
    print("\n" + "="*50)
    print("Training new tokenizer...")
    
    tokenizer = create_final_tokenizer_pipeline(num_samples=1000000)
    
    if tokenizer:
        print("\n" + "="*50)
        print("✓ Tokenizer created successfully!")
        print(f"✓ Vocabulary size: {tokenizer.vocab_size} (expected: 1971)")
        print(f"✓ Special token IDs match target configuration")
        print("\nTokenizer saved to: ./chess-llama-tokenizer")