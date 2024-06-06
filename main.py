import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter

# Step 2: Load the dataset
file_path = 'movies_metadata.csv'
df = pd.read_csv(file_path, low_memory=False)
print(df.head())

# Step 3: Explore the dataset
print(df.info())
print(df.isnull().sum())
print(df.describe())

# Step 4: Clean the dataset
df = df.dropna(subset=['budget', 'revenue', 'runtime', 'genres'])
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df = df.fillna(0)
print(df.info())

# Step 5: Analyze the data
top_rated_movies = df.nlargest(10, 'vote_average')[['title', 'vote_average']]
print(top_rated_movies)

def parse_genres(genres_str):
    genres_list = ast.literal_eval(genres_str)
    return [genre['name'] for genre in genres_list]

all_genres = df['genres'].apply(parse_genres).explode()
genre_counts = Counter(all_genres)
print(genre_counts.most_common(10))

correlation = df[['budget', 'revenue']].corr()
print(correlation)

# Step 6: Visualize the data
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# Plot 1: Top 10 highest-rated movies
sns.barplot(x='vote_average', y='title', data=top_rated_movies, ax=axes[0])
axes[0].set_title('Top 10 Highest-Rated Movies')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Movie Title')

# Plot 2: Most common genres
genres, counts = zip(*genre_counts.most_common(10))
sns.barplot(x=list(counts), y=list(genres), ax=axes[1])
axes[1].set_title('Most Common Genres')
axes[1].set_xlabel('Count')
axes[1].set_ylabel('Genre')

# Plot 3: Budget vs Revenue
sns.scatterplot(x='budget', y='revenue', data=df, ax=axes[2])
axes[2].set_title('Budget vs Revenue')
axes[2].set_xlabel('Budget')
axes[2].set_ylabel('Revenue')

# Adjust layout
plt.tight_layout()
plt.show()