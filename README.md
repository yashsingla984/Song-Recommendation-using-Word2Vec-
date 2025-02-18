# Song Recommendation System using Word2Vec

## Overview
This project implements a **song recommendation system** using **Word2Vec**, a popular embedding model. It trains on a dataset of playlists, learning relationships between songs based on their co-occurrence in playlists. Given a song ID, the model can recommend similar songs.

## Dataset
The dataset consists of:
- **Playlist Data:** A text file containing playlists, where each line represents a playlist with space-separated song IDs.
- **Song Metadata:** A mapping of song IDs to song titles and artists.

## How It Works
1. **Data Preparation:**
   - The dataset is loaded from an external URL.
   - Playlists with only one song are removed.
   - A DataFrame (`songs_df`) is created to store song metadata.

2. **Model Training:**
   - The Word2Vec model is trained on playlists, where each playlist is treated as a sequence of "words."
   - Parameters:
     - `vector_size=32`: Each song is represented by a 32-dimensional vector.
     - `window=20`: Context window of 20 songs is considered.
     - `negative=50`: Negative sampling technique is used with 50 negative examples.
     - `min_count=1`: All songs appearing at least once are included.
     - `workers=4`: Model is trained using 4 CPU threads.

3. **Generating Recommendations:**
   - Given a song ID, the model finds the most similar songs based on learned embeddings.
   - Results are displayed along with metadata (title and artist).

## Usage
### Dependencies
Ensure you have the following installed:
```bash
pip install pandas gensim numpy
