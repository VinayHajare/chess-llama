import requests
import os
import subprocess
import glob
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import chess.pgn
import io
import zipfile

class ChessDatasetProcessor:
    def __init__(self, output_dir="./chess_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download_lichess_database(self, start_year=2020, start_month=6, end_year=2025, end_month=11):
        """
        Download Lichess elite database for specified years and months from https://database.nikonoel.fr/
        Files are zip archives containing .pgn files.
        Download from June 2020 to November 2025
        """
        print("Downloading Lichess elite database...")
        base_url = "https://database.nikonoel.fr"
        raw_dir = os.path.join(self.output_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        for year in range(start_year, end_year + 1):
            month_start = start_month if year == start_year else 1
            month_end = end_month if year == end_year else 12

            for month in range(month_start, month_end + 1):
                filename = f"lichess_elite_{year}-{month:02d}.zip"
                url = f"{base_url}/{filename}"
                output_path = os.path.join(raw_dir, filename)

                print(f"Downloading {filename}...")

                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))

                with open(output_path, 'wb') as f, tqdm(
                    desc=filename,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        bar.update(size)

        print("Download complete!")

    def install_dependencies(self):
        """Install required system packages"""
        print("Installing dependencies...")
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "pgn-extract"], check=True)
        subprocess.run(["pip", "install", "chess", "tqdm", "requests"], check=True)

    def extract_and_filter_games(self):
        """
        Extract games from PGN files (inside zip) and filter for checkmate endings
        """
        raw_dir = os.path.join(self.output_dir, "raw")
        processed_dir = os.path.join(self.output_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        all_games = []

        for filepath in glob.glob(os.path.join(raw_dir, "*.zip")):
            print(f"Processing {os.path.basename(filepath)}...")

            # Extract zip file
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)

            # Find the extracted PGN file (assuming one per zip)
            pgn_path = filepath.replace('.zip', '.pgn')

            # Parse PGN and filter for checkmate games
            with open(pgn_path, 'r') as f:
                while True:
                    headers = chess.pgn.read_headers(f)
                    if headers is None:
                        break

                    termination = headers.get("Termination", "")
                    if "mate" in termination.lower():
                        f.seek(headers._offset)
                        game = chess.pgn.read_game(f)

                        if game:
                            uci_moves = []
                            board = game.board()

                            for move in game.mainline_moves():
                                uci_moves.append(move.uci())
                                board.push(move)

                            result = headers.get("Result", "0-1")
                            formatted_game = f"{result} " + " ".join(uci_moves)
                            all_games.append(formatted_game)

            # Clean up extracted file
            os.remove(pgn_path)

            # Save batch every 100k games
            if len(all_games) % 100000 == 0:
                batch_num = len(all_games) // 100000
                batch_file = os.path.join(processed_dir, f"games_batch_{batch_num}.txt")
                with open(batch_file, 'w') as f:
                    f.write('\n'.join(all_games[-100000:]))
                print(f"Saved batch {batch_num} with 100,000 games")

        # Save remaining games
        if all_games:
            final_file = os.path.join(processed_dir, "all_games.txt")
            with open(final_file, 'w') as f:
                f.write('\n'.join(all_games))
            print(f"Saved {len(all_games)} total games")

        return processed_dir

    def convert_to_uci_with_pgnextract(self):
        """
        Alternative method using pgn-extract for conversion
        """
        processed_dir = os.path.join(self.output_dir, "processed_uci")
        os.makedirs(processed_dir, exist_ok=True)

        raw_dir = os.path.join(self.output_dir, "raw")

        for filepath in glob.glob(os.path.join(raw_dir, "*.zip")):
            print(f"Converting {os.path.basename(filepath)}...")

            # Extract zip file
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)

            # Find the extracted PGN file
            pgn_path = filepath.replace('.zip', '.pgn')

            # Convert to UCI using pgn-extract
            output_file = os.path.join(processed_dir,
                                     os.path.basename(pgn_path).replace('.pgn', '.uci'))

            cmd = [
                "pgn-extract",
                "--notags",
                "--novars",
                "--nomovenumbers",
                "--noresults",
                "--uci",
                pgn_path
            ]

            with open(output_file, 'w') as f:
                subprocess.run(cmd, stdout=f, check=True)

            # Filter for games ending in checkmate
            self._filter_checkmate_games(output_file)

            os.remove(pgn_path)

    def _filter_checkmate_games(self, uci_file):
        """Filter UCI file for games ending in checkmate"""
        with open(uci_file, 'r') as f:
            games = f.read().strip().split('\n\n')

        filtered_games = []
        for game in games:
            moves = game.strip().split()
            if moves and moves[-1] in ['1-0', '0-1']:
                result = moves[-1]
                uci_moves = moves[:-1]
                formatted_game = f"{result} " + " ".join(uci_moves)
                filtered_games.append(formatted_game)

        # Overwrite with filtered games
        with open(uci_file, 'w') as f:
            f.write('\n'.join(filtered_games))

    def create_huggingface_dataset(self, processed_dir):
        """
        Create train/validation split and prepare for HuggingFace
        """
        from datasets import Dataset, DatasetDict
        import random

        all_games = []
        for filepath in glob.glob(os.path.join(processed_dir, "*.txt")):
            with open(filepath, 'r') as f:
                all_games.extend(line.strip() for line in f if line.strip())

        print(f"Total games: {len(all_games)}")

        random.shuffle(all_games)
        split_idx = int(len(all_games) * 0.95)

        train_games = all_games[:split_idx]
        val_games = all_games[split_idx:]

        train_dataset = Dataset.from_dict({"text": train_games})
        val_dataset = Dataset.from_dict({"text": val_games})

        dataset = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })

        dataset.save_to_disk(os.path.join(self.output_dir, "hf_dataset"))
        #dataset.push_to_hub(VinayHajare/chess-llama-dataset)
        return dataset

    def run_full_pipeline(self):
        """Run complete dataset processing pipeline"""
        print("Starting Chess Llama dataset pipeline...")
        self.install_dependencies()
        self.download_lichess_database()
        processed_dir = self.extract_and_filter_games()
        dataset = self.create_huggingface_dataset(processed_dir)
        print("Pipeline completed!")
        print(f"Train games: {len(dataset['train'])}")
        print(f"Validation games: {len(dataset['validation'])}")
        return dataset

if __name__ == "__main__":
    processor = ChessDatasetProcessor()
    processor.install_dependencies()
    processor.download_lichess_database()
    processor.extract_and_filter_games()
