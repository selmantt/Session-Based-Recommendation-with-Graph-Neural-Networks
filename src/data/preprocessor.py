
import os
import pickle
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class YoochoosePreprocessor:
    #Yoochoose dataset preprocessor
    
    def __init__(
        self,
        min_session_length=3,
        min_item_frequency=5,
        test_days=1
    ):
        #Initialize preprocessor
        self.min_session_length = min_session_length
        self.min_item_frequency = min_item_frequency
        self.test_days = test_days
        
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.n_items = 0
        
    def load_data(self, clicks_path):
        #Load raw click data
        print("[INFO] Loading raw click data...")
        
        df = pd.read_csv(
            clicks_path,
            header=None,
            names=["session_id", "timestamp", "item_id", "category"],
            dtype={
                "session_id": np.int64,
                "item_id": np.int64,
                "category": str
            },
            parse_dates=["timestamp"]
        )
        
        print(f"[INFO] Total clicks: {len(df):,}")
        print(f"[INFO] Total sessions: {df['session_id'].nunique():,}")
        print(f"[INFO] Total items: {df['item_id'].nunique():,}")
        
        return df
    
    def filter_by_item_frequency(self, df):
        #Filter rare items
        print(f"[INFO] Filtering items with < {self.min_item_frequency} occurrences...")
        
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= self.min_item_frequency].index
        df_filtered = df[df["item_id"].isin(valid_items)].copy()
        
        removed = len(df) - len(df_filtered)
        removed_pct = 100 * removed / len(df)
        print(f"[INFO] {removed:,} clicks removed ({removed_pct:.2f}%)")
        print(f"[INFO] Remaining items: {df_filtered['item_id'].nunique():,}")
        
        return df_filtered
    
    def filter_by_session_length(self, df):
        #Filter short sessions
        print(f"[INFO] Filtering sessions with < {self.min_session_length} clicks...")
        
        session_lengths = df.groupby("session_id").size()
        valid_sessions = session_lengths[session_lengths >= self.min_session_length].index
        df_filtered = df[df["session_id"].isin(valid_sessions)].copy()
        
        removed_sessions = df["session_id"].nunique() - df_filtered["session_id"].nunique()
        print(f"[INFO] {removed_sessions:,} sessions removed")
        print(f"[INFO] Remaining sessions: {df_filtered['session_id'].nunique():,}")
        
        return df_filtered
    
    def create_item_mapping(self, df):
        #Create item-index mapping
        unique_items = df["item_id"].unique()
        
        self.item_to_idx = {item: idx + 1 for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        self.n_items = len(unique_items) + 1
        
        print(f"[INFO] Mapping created for {self.n_items - 1} items (+ 1 padding)")
    
    def split_train_test(self, df):
        #Split data by time
        print(f"[INFO] Splitting data (last {self.test_days} days for test)...")
        
        max_date = df["timestamp"].max()
        split_date = max_date - timedelta(days=self.test_days)
        
        train_df = df[df["timestamp"] < split_date].copy()
        test_df = df[df["timestamp"] >= split_date].copy()
        
        print(f"[INFO] Train sessions: {train_df['session_id'].nunique():,}")
        print(f"[INFO] Test sessions: {test_df['session_id'].nunique():,}")
        
        return train_df, test_df
    
    def create_sessions(self, df):
        #Create session samples
        sessions = []
        grouped = df.groupby("session_id")
        
        for session_id, group in tqdm(grouped, desc="Creating sessions"):
            items = group.sort_values("timestamp")["item_id"].tolist()
            item_indices = [self.item_to_idx[item] for item in items]
            
            # Sequence augmentation
            for i in range(1, len(item_indices)):
                input_seq = item_indices[:i]
                target = item_indices[i]
                sessions.append((input_seq, target))
        
        return sessions
    
    def preprocess(self, clicks_path, data_fraction=1.0):
        #Run full preprocessing pipeline
        # Load data
        if data_fraction < 1.0:
            print(f"[INFO] Sampling {data_fraction*100:.2f}% of data...")
            
            print("[INFO] Scanning session IDs...")
            sessions_only = pd.read_csv(
                clicks_path, header=None, usecols=[0], names=["session_id"],
                dtype={"session_id": np.int64}
            )
            unique_sessions = sessions_only["session_id"].unique()
            del sessions_only
            
            n_sample = int(len(unique_sessions) * data_fraction)
            # Fixed seed for reproducible sampling - ensures checkpoint compatibility
            rng = np.random.RandomState(42)
            sampled_sessions = set(rng.choice(unique_sessions, n_sample, replace=False))
            del unique_sessions
            print(f"[INFO] {n_sample:,} sessions selected, loading chunks...")
            
            chunks = []
            for chunk in pd.read_csv(
                clicks_path, header=None, chunksize=500000,
                names=["session_id", "timestamp", "item_id", "category"],
                dtype={"session_id": np.int64, "item_id": np.int64, "category": str}
            ):
                filtered = chunk[chunk["session_id"].isin(sampled_sessions)]
                if len(filtered) > 0:
                    chunks.append(filtered)
            
            df = pd.concat(chunks, ignore_index=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            del chunks
            print(f"[INFO] {len(df):,} clicks loaded")
        else:
            df = self.load_data(clicks_path)
        
        # Iterative filtering
        prev_len = 0
        iteration = 0
        
        while len(df) != prev_len:
            prev_len = len(df)
            iteration += 1
            print(f"\n[INFO] Filtering iteration {iteration}")
            
            df = self.filter_by_item_frequency(df)
            df = self.filter_by_session_length(df)
        
        # Create item mapping
        self.create_item_mapping(df)
        
        # Split train/test
        train_df, test_df = self.split_train_test(df)
        
        # Create sessions
        print("\n[INFO] Creating train sessions...")
        train_sessions = self.create_sessions(train_df)
        
        print("\n[INFO] Creating test sessions...")
        test_sessions = self.create_sessions(test_df)
        
        # Summary
        print(f"\n[INFO] Final statistics:")
        print(f"  - Train samples: {len(train_sessions):,}")
        print(f"  - Test samples: {len(test_sessions):,}")
        print(f"  - Total items: {self.n_items:,}")
        
        return train_sessions, test_sessions
    
    def save(self, path):
        #Save preprocessor state
        state = {
            "item_to_idx": self.item_to_idx,
            "idx_to_item": self.idx_to_item,
            "n_items": self.n_items,
            "min_session_length": self.min_session_length,
            "min_item_frequency": self.min_item_frequency
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        print(f"[INFO] Preprocessor state saved: {path}")
    
    @classmethod
    def load(cls, path):
        #Load preprocessor state
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        preprocessor = cls(
            min_session_length=state["min_session_length"],
            min_item_frequency=state["min_item_frequency"]
        )
        preprocessor.item_to_idx = state["item_to_idx"]
        preprocessor.idx_to_item = state["idx_to_item"]
        preprocessor.n_items = state["n_items"]
        
        print(f"[INFO] Preprocessor state loaded: {path}")
        return preprocessor


# Helper functions
def save_sessions(sessions, path):
    #Save processed sessions
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(sessions, f)
    
    print(f"[INFO] {len(sessions):,} sessions saved: {path}")


def load_sessions(path):
    #Load processed sessions
    with open(path, "rb") as f:
        sessions = pickle.load(f)
    
    print(f"[INFO] {len(sessions):,} sessions loaded: {path}")
    return sessions
