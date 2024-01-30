import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Simulated data for demonstration
song_data = {
    'SongID': [1, 2, 3],
    'Title': ['Song A', 'Song B', 'Song C'],
    'AudioFilePath': ['/path/to/songA.mp3', '/path/to/songB.mp3', '/path/to/songC.mp3'],
}

user_data = {
    'UserID': [101, 102, 103, 104, 105],
    'SongID': [1, 1, 2, 2, 3],
    'ListenCount': [10, 15, 5, 8, 12],
}

# Create DataFrames
songs_df = pd.DataFrame(song_data)
users_df = pd.DataFrame(user_data)

def extract_song_features(audio_file_path):
    y, sr = librosa.load(audio_file_path)
    tempo, key = librosa.beat.beat_track(y, sr)
    energy = np.mean(librosa.feature.rms(y))
    return tempo, key, energy

# Extract song features
songs_df[['Tempo', 'Key', 'Energy']] = songs_df['AudioFilePath'].apply(lambda x: pd.Series(extract_song_features(x)))

# Merge with streaming data
merged_df = pd.merge(users_df, songs_df, on='SongID')

# User segmentation based on listening behaviors
user_segments = pd.cut(merged_df['ListenCount'], bins=[0, 5, 10, 15, np.inf], labels=['Novice', 'Casual', 'Regular', 'Superfan'])

# Add user segments to the DataFrame
merged_df['UserSegment'] = user_segments

# Display the merged DataFrame
print(merged_df)

# Plot features for each song
for index, row in songs_df.iterrows():
    plt.figure(figsize=(8, 4))
    y, sr = librosa.load(row['AudioFilePath'])
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform for {row['Title']}")
    plt.show()
