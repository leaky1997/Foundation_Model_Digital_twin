from data_base import ClassificationDataset,\
    AnomalyDetectionDataset,\
    ImputationDataset,\
    ForecastingDataset
    
    
class THU_006_classification(ClassificationDataset):
    def __init__(self, args, flag):
        super().__init__(args, flag)


        
class THU_006_Forecasting(ForecastingDataset):
    def __init__(self, args, flag):
        super().__init__(args, flag)


