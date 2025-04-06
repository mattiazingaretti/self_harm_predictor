import pandas as pd
from models.SelfHarmDatasetsContainer import SelfHarmDatasetsContainer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

"""
    This service is responsible for merging the preprocessed datasets and preparing them for training.
"""
class TrainingDataProcessorService:

    def __init__(self, df_container: SelfHarmDatasetsContainer):
        self.df_container = df_container
        self.combined_data = self._merge_data_set()

    def _merge_data_set(self):
        self.df_container.mental_health_data['label'] = (self.df_container.mental_health_data['status'] == 'Suicidal').astype(int)
        self.df_container.suicidal_tweet_detection_dataset['label'] = (self.df_container.suicidal_tweet_detection_dataset['Suicide'] == 'Potential Suicide post').astype(int)

        combined_data = pd.concat([
            self.df_container.mental_health_data[['statement', 'label']].rename(columns={'statement': 'text'}),
            self.df_container.suicidal_tweet_detection_dataset[['Tweet', 'label']].rename(columns={'Tweet': 'text'})
        ], ignore_index=True)

        return combined_data.dropna()
    

    def split_dataset(self, 
                        max_features=5000,
                        stop_words='english',
                        test_size=0.2,
                        random_state=42 ):
        tfidf = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
        X = tfidf.fit_transform(self.combined_data['text']).toarray()
        y = self.combined_data['label'].values

        return train_test_split(X, y, test_size=test_size, random_state=random_state)