import pandas as pd

# TODO: Determine whether you are evaluating Experiment A or Experiment B here
EXPERIMENT = "A"

if EXPERIMENT == A:
  
  # TODO: Replace with your model's evaluation CSV file (by Claude or GPT-4o) for Experiment A
  csv_path = '/path/to/evaluation/file.csv'
  df = pd.read_csv(csv_path)
  
  # Group by 'type' and calculate the average rating
  average_scores = df.groupby('type')['rating'].mean()
  
  # Print the results
  print("Average scores by type:")
  print(average_scores)

else:
  # TODO: Replace with your model's evaluation CSV file (by Claude or GPT-4o) for Experiment B
  csv_path = '/path/to/evaluation/file.csv'  # replace with your actual path
  df = pd.read_csv(csv_path)
  
  # Group by 'type' and calculate the average rating
  average_graph_scores = df.groupby('type')['graph_rating'].mean()
  average_response_scores = df.groupby('type')['response_rating'].mean()
  
  # Print the results
  print("Average graph scores by type:")
  print(average_graph_scores)
  print("Average response scores by type:")
  print(average_response_scores)
