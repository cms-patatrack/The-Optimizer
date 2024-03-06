import numpy as np

class FeatureImportance:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def compute_correlation(self):
        history = self.optimizer.history
        # i.e. history = [[{'position': [0.1, 0.2, 0.3], 'fitness': [0.1, 0.2]}, {'position': [0.2, 0.3, 0.4], 'fitness': [0.2, 0.3]}],
        #                [{'position': [0.1, 0.2, 0.3], 'fitness': [0.1, 0.2]}, {'position': [0.2, 0.3, 0.4], 'fitness': [0.2, 0.3]}]]
        correlations = np.zeros((len(history[0][0]['position']), len(history[0][0]['fitness'])))
        for p_id in range(len(history[0][0]['position'])):
            for f_id in range(len(history[0][0]['fitness'])):
                correlation = np.corrcoef([history[i][j]['position'][p_id] for i in range(len(history)) for j in range(len(history[i]))],
                                            [history[i][j]['fitness'][f_id] for i in range(len(history)) for j in range(len(history[i]))])
                correlations[p_id, f_id] = correlation[0][1]
        return correlations
        
                