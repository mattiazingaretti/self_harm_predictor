from services.DataLoaderService import DataLoaderService
from services.PreprocessorService import PreprocessorService


class TrainingPipelineService:

    def __init__(self, data_loader_service: DataLoaderService = DataLoaderService() ):
        self.df = data_loader_service.load_all_datasets()

    def train_model(self):
        preprocessor_service = PreprocessorService(self.df)
        