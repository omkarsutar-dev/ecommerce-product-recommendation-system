import numpy as np
import pickle

class MFRecommender:
    def __init__(self, model_path, user_map, item_map, reverse_item_map):
        self.P, self.Q = pickle.load(open(model_path, "rb"))
        self.user_map = user_map
        self.item_map = item_map
        self.reverse_item_map = reverse_item_map

    def recommend(self, user_id, k=10):
        if user_id not in self.user_map:
            return []

        uidx = self.user_map[user_id]
        scores = np.dot(self.Q, self.P[uidx])
        top_items = np.argsort(scores)[::-1][:k]

        return [self.reverse_item_map[i] for i in top_items]
