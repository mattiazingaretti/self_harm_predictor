class SelfHarmDatasetsContainer:

    def __init__(self , mental_health_data, reddit_mental_health_data, suicidal_tweet_detection_dataset, dreaddit):
        if (mental_health_data is None or reddit_mental_health_data is None or suicidal_tweet_detection_dataset is None or dreaddit is None):
            raise ValueError("All datasets must be provided")
        self.mental_health_data = mental_health_data
        self.reddit_mental_health_data = reddit_mental_health_data
        self.suicidal_tweet_detection_dataset = suicidal_tweet_detection_dataset
        self.dreaddit = dreaddit 
        
    def update_mental_health_data(self, new_data):
        if new_data is None:
            raise ValueError("New data must be provided")
        self.mental_health_data = new_data
    
    def update_reddit_mental_health_data(self, new_data):
        if new_data is None:
            raise ValueError("New data must be provided")
        self.reddit_mental_health_data = new_data

    def update_suicidal_tweet_detection_dataset(self, new_data):
        if new_data is None:
            raise ValueError("New data must be provided")
        self.suicidal_tweet_detection_dataset = new_data
    
    def update_dreaddit(self, new_data):
        if new_data is None:
            raise ValueError("New data must be provided")
        self.dreaddit = new_data