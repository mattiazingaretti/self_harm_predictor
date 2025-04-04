import UtilityService
from models.SelfHarmDatasetsContainer import SelfHarmDatasetsContainer

class DataLoaderService:
    def __init__(self):
        pass
        
    def load_mental_health_dataset(self):
        return UtilityService.load_kaggle_dataset(
            "szegeelim/mental-health",
            "Combined Data.csv"
        )

    def load_reddit_dataset(self):
        return UtilityService.load_kaggle_dataset(
            "neelghoshal/reddit-mental-health-data",
            "data_to_be_cleansed.csv"
        )
    
    def load_suicidal_tweet_dataset(self):
        return UtilityService.load_kaggle_dataset(
            "aunanya875/suicidal-tweet-detection-dataset",
            "data_to_be_cleansed.csv"
        )
    
    def load_dreaddit_dataset(self):
        return UtilityService.load_dataset_from_zip("http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip")
    

    def load_all_datasets(self):
        return SelfHarmDatasetsContainer(
          mental_health_data=self.load_mental_health_dataset(),
          reddit_mental_health_data=self.load_reddit_dataset(),
          suicidal_tweet_detection_dataset=self.load_suicidal_tweet_dataset(),
          dreaddit=self.load_dreaddit_dataset()
        )  