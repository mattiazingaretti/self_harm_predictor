from models.SelfHarmDatasetsContainer import SelfHarmDatasetsContainer


class PreprocessorService:
    
    def __init__(self, df_container :SelfHarmDatasetsContainer):
        if df_container is None:
            raise ValueError("Data container must be provided")
        self.df_container = df_container

    def preprocess_mental_health(self):
        self.df_container.mental_health_data = self.df_container.mental_health_data.dropna()
        self.df_container.mental_health_data = self.df_container.mental_health_data.drop_duplicates(subset='statement')
        return self.df_container
    
    # See FIXME in DataLoaderService.py
    def preprocess_reddit_dataset(self):
        self.df_container.reddit_mental_health_data['target'] = self.df_container.reddit_mental_health_data['target'].map({0: 'Stress', 1:'Depression', 2: 'Bipolar disorder', 3: 'Personality disorder', 4: 'Anxiety'})
        self.df_container.reddit_mental_health_data = self.df_container.reddit_mental_health_data.dropna()
        self.df_container.reddit_mental_health_data = self.df_container.reddit_mental_health_data.drop_duplicates(subset='text')
        self.df_container.reddit_mental_health_data['social'] = 'Reddit'
        return self.df_container

    def preprocess_suicidal_tweet_dataset(self):
        self.df_container.suicidal_tweet_detection_dataset = self.df_container.suicidal_tweet_detection_dataset.dropna()
        self.df_container.suicidal_tweet_detection_dataset = self.df_container.suicidal_tweet_detection_dataset.drop_duplicates(subset='Tweet')
        self.df_container.suicidal_tweet_detection_dataset['social'] = 'Twitter'
        return self.df_container

    def preprocess_dreaddit_dataset(self):
        self.df_container.dreaddit['label'] = self.df_container.dreaddit['label'].map({0: 'Not stressful', 1: 'Stressful'})
        self.df_container.dreaddit = self.df_container.dreaddit.loc[self.df_container.dreaddit['text'] != '#NAME?']
        self.df_container.dreaddit = self.df_container.dreaddit.drop_duplicates(subset=['subreddit', 'text'])
        self.df_container.dreaddit['social'] = 'Reddit'
        return self.df_container
    
    def preprocess_all(self):
        self.preprocess_mental_health()
        # self.preprocess_reddit_dataset() # SEE FIXME IN DataLoaderService.py
        self.preprocess_suicidal_tweet_dataset()
        self.preprocess_dreaddit_dataset()
        return self.df_container