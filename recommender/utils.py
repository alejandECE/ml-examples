#  Created by Luis Alejandro (l.alejandro.2011@gmail.com).
#  Copyright © Do not distribute or use without authorization from author

from typing import Tuple
import numpy as np
import pathlib
import pandas as pd
import os
import argparse
from tqdm import tqdm

DATASETS = pathlib.Path(os.environ['DATASETS'])

ROOT = pathlib.Path(pathlib.Path(__file__).parent)


def get_unique_movies_ids(ratings: pd.DataFrame) -> np.ndarray:
  movies = ratings['movieId'].unique()
  print(f"Total rated movies: {movies.shape[0]}, indexes range from {movies.min()} to {movies.max()}")
  return movies


def get_unique_users_ids(ratings: pd.DataFrame) -> np.ndarray:
  users = ratings['userId'].unique()
  print(f"Total users with ratings: {users.shape[0]}, indexes range from {users.min()} to {users.max()}")
  return users


# Process raw ratings by creating contiguous indexes for movies and users
def process_raw_ratings_from_csv(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
  # Creates a mapping from original movie indexes to new (contiguous) indexes
  print('Before processing!')
  movies = get_unique_movies_ids(ratings)
  movie_to_index = {movieId: index for index, movieId in enumerate(sorted(movies))}

  # Creates a mapping from original user indexes to new (contiguous) indexes
  users = get_unique_users_ids(ratings)
  user_to_index = {userId: index for index, userId in enumerate(sorted(users))}

  # Transform dataframe
  ratings['userId'] = ratings['userId'].map(user_to_index)
  ratings['movieId'] = ratings['movieId'].map(movie_to_index)

  # Verifies mapping
  print('After processing!')
  get_unique_movies_ids(ratings)
  get_unique_users_ids(ratings)

  # Returns dataframe
  return ratings, user_to_index, movie_to_index


def load_mapping_to_index_from(textfile) -> dict:
  with open(textfile) as file:
    mapping = {int(entry.strip()): index for index, entry in enumerate(file.readlines())}
  return mapping


def save_index_mapping_to(textfile, mapping) -> None:
  with open(textfile, 'w') as file:
    file.writelines([f'{key}\n' for key, _ in sorted(mapping.items(), key=lambda entry: entry[1])])


def split_at_random(ratings: pd.DataFrame, test_pct: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # Shuffling once (to speed up things)! It does modify original dataframe but during training this is shuffle again so!
  ratings = ratings.sample(frac=1)
  # Find unique users, and counts of ratings per user
  unique_users, indices, unique_counts = np.unique(ratings['userId'], return_inverse=True, return_counts=True)
  rows_per_user = np.split(np.argsort(indices), np.cumsum(unique_counts)[:-1])
  train = []
  test = []
  for i in tqdm(range(unique_users.shape[0])):
    samples = int(unique_counts[i] * test_pct)
    train.extend(rows_per_user[i][:-samples])
    test.extend(rows_per_user[i][-samples:])

  return ratings.iloc[train, :], ratings.iloc[test, :]


def split_by_timestamp(ratings: pd.DataFrame, leave_out: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
  ratings.sort_values(by=['userId', 'timestamp'], inplace=True)
  # Find unique users, and counts of ratings per user
  unique_users, unique_counts = np.unique(ratings['userId'], return_counts=True)
  rows_per_user = np.split(np.arange(len(ratings)), np.cumsum(unique_counts)[:-1])
  train = []
  test = []
  for i in tqdm(range(unique_users.shape[0])):
    train.extend(rows_per_user[i][:-leave_out])
    test.extend(rows_per_user[i][-leave_out:])

  return ratings.iloc[train, :], ratings.iloc[test, :]


def split_by_sequence(ratings: pd.DataFrame, length: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # Activates tqdm for pandas
  tqdm.pandas()

  # Sorts by timestamp
  ratings.sort_values(by='timestamp', inplace=True, ascending=True)

  # Groups by user
  groups = ratings.groupby(by='userId', group_keys=False)

  def generate_sequences(user_data):
    user = user_data['userId'].values[0]
    movies = user_data['movieId'].values
    if len(movies) < length + 1:
      return
    indices = np.repeat(np.array(range(length + 1), ndmin=2), repeats=len(movies) - length, axis=0)
    indices += np.expand_dims(np.array(range(len(movies) - length)), axis=-1)
    sequences = np.take(movies, indices, axis=0)
    return pd.DataFrame(
      data=np.concatenate((np.expand_dims(np.repeat(user, len(movies) - length), axis=-1), sequences), axis=1),
      columns=['userId'] + [str(i) for i in range(length)] + ['target']
    )

  # Builds sequences
  sequence_df = ratings.groupby(by='userId', group_keys=False).progress_apply(generate_sequences)

  # Splits
  groups = sequence_df.groupby(by='userId', group_keys=False)
  train_df = groups.progress_apply(lambda data: data.iloc[:-1, :])
  test_df = groups.progress_apply(lambda data: data.iloc[-1, :])

  return train_df, test_df


# Loads ratings from the proper csv file (creates splitting if does not exist)
def load_ratings_20m(split: str = 'random', test_pct=5e-4) -> Tuple:
  # Splits folder
  folder = ROOT / f'data/ml-20m/splits/{split}'
  # If files don't exist create them
  if not folder.exists():
    folder.mkdir(parents=True)
    # Preprocess original ratings file
    ratings, user_to_index, movie_to_index = process_raw_ratings_from_csv(
      pd.read_csv(DATASETS / 'recommender/movies/ml-20m/ratings.csv')
    )
    # Stores mappings
    save_index_mapping_to(folder / 'user_to_index.txt', user_to_index)
    save_index_mapping_to(folder / 'movie_to_index.txt', movie_to_index)

    # Splits
    if split == 'random':
      train_df, test_df = split_at_random(ratings, test_pct)
    elif split == 'timestamp':
      train_df, test_df = split_by_timestamp(ratings)
    elif split == 'sequence':
      train_df, test_df = split_by_sequence(ratings)
    else:
      raise Exception('Split method not recognized')

    # Stores
    train_df.to_csv(folder / 'train_df.csv', index=False)
    test_df.to_csv(folder / 'test_df.csv', index=False)

  # Loads titles
  movies = pd.read_csv(DATASETS / 'recommender/movies/ml-20m/movies.csv', index_col='movieId')
  movie_to_title = movies.to_dict()

  # Loads info from corresponding files
  return (
    pd.read_csv(folder / 'train_df.csv'),
    pd.read_csv(folder / 'test_df.csv'),
    load_mapping_to_index_from(folder / 'user_to_index.txt'),
    load_mapping_to_index_from(folder / 'movie_to_index.txt'),
    movie_to_title
  )


# Loads ratings from the proper csv file (creates splitting if does not exist)
def load_ratings_100k(split: str = 'random', test_pct=1e-1) -> Tuple:
  # Splits folder
  folder = ROOT / f'data/ml-100k/splits/{split}'
  # If files don't exist create them
  if not folder.exists():
    folder.mkdir(parents=True)
    # Preprocess original ratings file
    ratings, user_to_index, movie_to_index = process_raw_ratings_from_csv(
      pd.read_csv(DATASETS / 'recommender/movies/ml-100k/u.data',
                  sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'], encoding='latin-1')
    )

    # Stores mappings
    save_index_mapping_to(folder / 'user_to_index.txt', user_to_index)
    save_index_mapping_to(folder / 'movie_to_index.txt', movie_to_index)

    # Splits
    if split == 'random':
      train_df, test_df = split_at_random(ratings, test_pct)
    elif split == 'timestamp':
      train_df, test_df = split_by_timestamp(ratings)
    elif split == 'sequence':
      train_df, test_df = split_by_sequence(ratings)
    else:
      raise Exception('Split method not recognized')

    # Stores
    train_df.to_csv(folder / 'train_df.csv', index=False)
    test_df.to_csv(folder / 'test_df.csv', index=False)

  # Loads titles
  genres_df = pd.read_csv(DATASETS / 'recommender/movies/ml-100k/u.genre', sep='|', names=['genre', 'id'],
                          encoding='latin-1')
  genres = list(genres_df['genre'])
  columns = ['movie_id', 'title', 'release_date', "video_release_date", "imdb_url"] + genres
  movies_df = pd.read_csv(DATASETS / 'recommender/movies/ml-100k/u.item',
                          sep='|',
                          names=columns,
                          index_col='movie_id', encoding='latin-1')
  movie_to_title = movies_df['title'].to_dict()

  # Loads info from corresponding files
  return (
    pd.read_csv(folder / 'train_df.csv'),
    pd.read_csv(folder / 'test_df.csv'),
    load_mapping_to_index_from(folder / 'user_to_index.txt'),
    load_mapping_to_index_from(folder / 'movie_to_index.txt'),
    movie_to_title
  )


if __name__ == '__main__':
  version_to_function = {
    'ml-20m': load_ratings_20m,
    'ml-100k': load_ratings_100k
  }
  versions = ', '.join(version_to_function.keys())
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset', help=f'Dataset to load: {versions}', type=str)
  args = parser.parse_args()
  option = args.dataset
  if option in version_to_function:
    dataset, _, _ = version_to_function[option]()
    print(dataset.head(20))
  else:
    print(f'Dataset version not supported! Choose one of: {versions}')
