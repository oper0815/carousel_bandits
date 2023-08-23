from functools import reduce
from scipy.special import expit
import numpy as np


# This class uses user and playlist features datasets to simulate users responses to a list of recommendations
class ContextualEnvironment():
    def __init__(self, user_features, playlist_features, user_segment, n_recos):
        self.user_features = user_features
        self.playlist_features = playlist_features
        self.user_segment = user_segment
        self.n_recos = n_recos
        self.th_segment_rewards = np.zeros(user_features.shape[0])  # user_features.shape[0] = the number of users in user_features dataset -> segment reward
        self.th_rewards = np.zeros(user_features.shape[0])          # user_features.shape[0] = the number of users in user_features dataset -> user reward
        self.compute_optimal_theoretical_rewards()
        self.compute_segment_optimal_theoretical_rewards()

    # Computes expected reward for each user given their recommendations
    def compute_theoretical_rewards(self, batch_user_ids, batch_recos):
        batch_user_features = np.take(self.user_features, batch_user_ids, axis = 0)         # batch 안에 있는 user에 대한 user_feature만 추출 / (9,97)
        batch_playlist_features = np.take(self.playlist_features, batch_recos, axis = 0)    # 추천리스트에 있는 item에 대한 feature 추출 > (user 수, 추천 item 수, item feature dim)
        n_users = len(batch_user_ids)                                                       # batch 내에 있는 유저 수
        th_reward = np.zeros(n_users)                                                       # zero vector 생성 (user 수)
        for i in range(n_users):                                                            # user 수 만큼 for 문 수행
            probas = expit(batch_user_features[i].dot(batch_playlist_features[i].T))        # 선호도 > sigmoid of (한 user의 dim_user_feature) x T(추천 item 수 x dim_item_feature) / batch_playlist_features[i] : i 번째 사람의 추천 item list에 대한 feature
            th_reward[i] = 1 - reduce(lambda x,y : x * y, 1 - probas)                       # batch n-th user의 reward > 확률의 곱 ??? why ?
        return th_reward

    # Computes list of n recommendations with highest expected reward for each user
    def compute_optimal_recos(self, batch_user_ids, n):
        batch_user_features = np.take(self.user_features, batch_user_ids, axis = 0)     # batch 안에 있는 user에 대한 user_feature만 추출
        n_users = len(batch_user_ids)                                                   # batch 내에 있는 유저 수
        probas = batch_user_features.dot(self.playlist_features.T)                      # (user 수 x dim_feature) x T(item 수 x dim_feature) => user x item matrix
        optim = np.argsort(-probas)[:, :n]                                              # argsort는 작은 것 부터 정렬 -> (-) 표기로 내림차순 -> n개만 추출
        return optim                                                                    # return : batch user에 대한 top N 추천리스트 ITEM INDEX

    # Computes highest expected reward for each user
    def compute_optimal_theoretical_rewards(self):
        n_users = self.user_features.shape[0]
        u = 0
        step = 100000
        while u < n_users:
            users_ids = range(u, min(n_users, u + step))                            # range(0, 100000) / range(100000, 200000) > min 함수는 마지막 배치를 위해
            opt_recos = self.compute_optimal_recos(users_ids, self.n_recos)         # compute_optimal_recos 수행 (batch user id, 추천 item 수)
            opt_rewards = self.compute_theoretical_rewards(users_ids, opt_recos)    # compute_theoretical_rewards 수행 (batch user id, batch 유저의 추천 list)
            self.th_rewards[u:min(n_users, u + step)] = opt_rewards                 # batch reward를 opt_rewards[batch] 에 저장
            u += step                                                               # batch change
        return 

    # Computes list of n recommendations with highest expected reward for each segment
    def compute_segment_optimal_recos(self, n):
        n_segments = len(np.unique(self.user_segment))                              # unique 한 segment 수
        segment_recos = np.zeros((n_segments, n), dtype = np.int64)                 # (segment 수, 추천리스트 수) > segment 별로 동일한 추천리스트 생성
        for i in range(n_segments):
            mean_probas = np.mean(expit(np.take(self.user_features, np.where(self.user_segment == i)[0], axis = 0).dot(self.playlist_features.T)), axis = 0)
            reward = 1 - reduce(lambda x,y : x * y, 1 + np.sort(-mean_probas)[:n])
            segment_recos[i] = np.argsort(-mean_probas)[:n]
        return segment_recos                                                        # segment 별 추천리스트 [[seg1 reclist], [seg2 reclist] ...]
    
    # Computes highest expected reward for each segment
    def compute_segment_optimal_theoretical_rewards(self):
        n_users = self.user_features.shape[0]
        u = 0
        step = 100000
        segment_recos = self.compute_segment_optimal_recos(self.n_recos)            # 추천리스트 추출 (배치별)        
        while u < n_users:
            users_ids = range(u, min(n_users, u+ step))
            user_segment = np.take(self.user_segment, users_ids)                    # batch 안에 있는 user의 segment 추출
            opt_recos = np.take(segment_recos, user_segment, axis = 0)              # segment 별 추천리스트 매핑 (길이는 batch 수) [[1,2,3], [1,2,3] ...]
            opt_rewards = self.compute_theoretical_rewards(users_ids, opt_recos)    # compute_theoretical_rewards 수행 (batch user id, batch 유저의 추천 list)
            self.th_segment_rewards[u:min(n_users, u+ step)] = opt_rewards          # batch reward를 opt_rewards[batch] 에 저장
            u += step
        return 

    # Given a list of users and their respective list of recos (each of size self.n_recos), computes
    # corresponding simulated reward
    def simulate_batch_users_reward(self, batch_user_ids, batch_recos):
        
        # First, compute probability of streaming each reco and draw rewards accordingly
        batch_user_features = np.take(self.user_features, batch_user_ids, axis = 0)
        batch_playlist_features = np.take(self.playlist_features, batch_recos, axis = 0)
        n_users = len(batch_user_ids)
        n = len(batch_recos[0])
        probas = np.zeros((n_users, n))
        for i in range(n_users):
            probas[i] = expit(batch_user_features[i].dot(batch_playlist_features[i].T)) # probability to stream each reco
        rewards = np.zeros((n_users, n))
        i = 0
        rewards_uncascaded = np.random.binomial(1, probas) # drawing rewards from probabilities
        positive_rewards = set()

        # Then, for each user, positive rewards after the first one are set to 0 (and playlists as "unseen" subsequently)
        # to imitate a cascading browsing behavior
        # (nonetheless, users can be drawn several times in the batch of a same round ; therefore, each user
        # can have several positive rewards - i.e. stream several playlists - in a same round, consistently with
        # the multiple-plays framework from the paper)
        nz = rewards_uncascaded.nonzero()
        for i in range(len(nz[0])):
            if nz[0][i] not in positive_rewards:
                rewards[nz[0][i]][nz[1][i]] = 1
                positive_rewards.add(nz[0][i])
        return rewards
