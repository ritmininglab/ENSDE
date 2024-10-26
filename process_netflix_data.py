import os
import pandas as pd

def process_netflix_data(data_dir):
    data = []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'r') as file:
            movie_id = None
            for line in file:
                if line.endswith(':\n'):
                    # This is a movie ID line
                    movie_id = line.strip()[:-1]
                else:
                    # This is a user rating line
                    user_id, rating, date = line.strip().split(',')
                    data.append([user_id, movie_id, rating, date])

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['user', 'item', 'rating', 'time'])

    # Convert data types
    df['user'] = df['user'].astype(int)
    df['item'] = df['item'].astype(int)
    df['rating'] = df['rating'].astype(int)
    df['time'] = pd.to_datetime(df['time'])

    return df

# Specify the directory containing the Netflix Prize dataset files
data_path = os.getcwd()
data_dir = data_path+'/netflix'

# Process the dataset
df = process_netflix_data(data_dir)
df.to_pickle(data_dir+'/processed_netflix_data_all_users.pkl')
df = df.head(100)
# Save the processed data to a CSV file
df.to_csv(data_dir+'/processed_netflix_data_100_users.csv', index=False)

print("Processing complete. Data saved to 'processed_netflix_data.csv'.")