from fastapi import FastAPI, Request
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def _entropy(self, y):
        unique_classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _information_gain(self, y, y_left, y_right):
        parent_entropy = self._entropy(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        weighted_entropy = (n_left / n) * self._entropy(y_left) + (n_right / n) * self._entropy(y_right)
        gain = parent_entropy - weighted_entropy
        return gain

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                y_left = y[X[:, feature_idx] <= threshold]
                y_right = y[X[:, feature_idx] > threshold]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gain = self._information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        unique_classes, counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]

        if len(unique_classes) == 1 or depth == self.max_depth:
            return {"leaf": True, "class": majority_class}

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {"leaf": True, "class": majority_class}

        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        left_tree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree
        }

    def fit(self, X, y):
        self.tree = self._build_tree(np.array(X), np.array(y))

    def _predict(self, x, tree):
        if tree["leaf"]:
            return tree["class"]

        feature = tree["feature"]
        threshold = tree["threshold"]
        if x[feature] <= threshold:
            return self._predict(x, tree["left"])
        else:
            return self._predict(x, tree["right"])

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in np.array(X)])





# Khởi tạo app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://anime-fawn-five.vercel.app"],  # Cho phép tất cả origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối đến MongoDB.
client = MongoClient('mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/anime_tango2')  # Sử dụng URL kết nối MongoDB của bạn
db = client['anime_tango2']  # Tên cơ sở dữ liệu của bạn
anime_collection = db['Anime']
user_rating_collection = db['UserRating']

# Hàm lấy dữ liệu Anime
def get_anime_data():
    anime_data = list(anime_collection.find())
    return pd.DataFrame(anime_data)

# Hàm lấy dữ liệu UserRatings (thay vì UserFavorites)
def get_user_ratings(user_id):
    user_ratings = list(user_rating_collection.find({'User_id': user_id}))
    return user_ratings

# Lấy dữ liệu Anime
anime_df = get_anime_data()
anime_df2 = anime_df
# Cập nhật để phân loại cột 'Score' theo các điều kiện
def categorize_score(score):
    if score < 8:
        return 0  # Loại 0: Score < 8
    elif 8 <= score <= 9:
        return 1  # Loại 1: 8 <= Score <= 9
    else:
        return 2  # Loại 2: Score >= 9

# Thêm cột 'Type' dựa trên cột 'Score'
anime_df['Score_'] = anime_df['Score'].apply(categorize_score)

# Chuyển Genres thành các cột nhị phân (one-hot encoding)
genres = ['Action', 'Adventure','Avant Garde','Award Winning','Ecchi','Girls Love','Mystery','Sports','Supernatural','Suspense', 'Sci-Fi', 'Comedy', 'Drama', 'Romance', 'Horror', 'Fantasy', 'Slice of Life']
for genre in genres:
    anime_df[genre] = anime_df['Genres'].apply(lambda x: 1 if genre in x else 0)

# Thêm cột 'Favorites' dựa trên số lượng Favorites
def categorize_favorites(favorites_count):
    if favorites_count <= 5000:
        return 0  # Thấp
    elif favorites_count <= 20000:
        return 1  # Trung bình
    else:
        return 2  # Cao

anime_df['Favorites_'] = anime_df['Favorites'].apply(categorize_favorites)

# Thêm cột 'JapaneseLevel' từ Anime
def categorize_japanese_level(level):
    if level in ['N4', 'N5']:  # Các mức độ dễ học
        return 0
    elif level in ['N2', 'N3']:  # Các mức độ dễ học
        return 1
    else :
        return 2

anime_df['JapaneseLevel_'] = anime_df['JapaneseLevel'].apply(categorize_japanese_level)

# Cập nhật phần 'Is_13_plus' và thêm các độ tuổi khác
def categorize_age(age_str):
    if '7+' in age_str:
        return 0  # Các anime có độ tuổi 13+
    elif '13+' in age_str:
        return 1  # Các anime có độ tuổi 13+
    elif '16+' in age_str:
        return 2  # Các anime có độ tuổi 16+
    elif '17+' in age_str:
        return 3  # Các anime có độ tuổi 17+
    elif '18+' in age_str:
        return 4  # Các anime có độ tuổi 18+
    else:
        return 0  # Các anime không có độ tuổi

anime_df['AgeCategory'] = anime_df['Old'].apply(categorize_age)

def get_user_features(user_ratings_df, anime_df, threshold=10):
    """
    Trích xuất đặc trưng người dùng dựa trên lịch sử đánh giá.
    Nếu người dùng có ít hơn threshold đánh giá, lấy trung bình từ toàn bộ anime.
    """
    if user_ratings_df.empty or len(user_ratings_df) < threshold:
        features = anime_df[['Old', 'Favorites_', 'JapaneseLevel_', 'AgeCategory', 'Score_']].mean(axis=0).to_dict()
    else:
        features = {
            'Avg_Old': user_ratings_df['Old'].mean(),
            'Avg_Favorites': user_ratings_df['Favorites_'].mean(),
            'Avg_JapaneseLevel': user_ratings_df['JapaneseLevel_'].mean(),
            'Avg_AgeCategory': user_ratings_df['AgeCategory'].mean(),
            'Avg_Score': user_ratings_df['Score_'].mean(),
        }
    return features

def train_decision_tree(user_id):
    # Lấy đặc trưng của người dùng
    user_features = get_user_features(user_id)
    user_ratings = get_user_ratings(user_id)
    rated_anime_ids = [rating['Anime_id'] for rating in user_ratings]

    # Tạo tập dữ liệu
    anime_features = anime_df[genres + ['Favorites_', 'JapaneseLevel_', 'AgeCategory', 'Score_']]
    X = anime_features.values
    y = np.array([1 if anime_id in rated_anime_ids else 0 for anime_id in anime_df['Anime_id']])

    # Huấn luyện mô hình Decision Tree
    clf = DecisionTree(max_depth=3)
    clf.fit(X, y)

    return clf

@app.post('/decisiontree')
async def recommend_anime(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    n = data.get("n", 10)  # Số lượng gợi ý, mặc định là 10
    clf = train_decision_tree(user_id)
    anime_features = anime_df[genres + ['Favorites_', 'JapaneseLevel_', 'AgeCategory', 'Score_']]
    predictions = clf.predict(anime_features)

    recommended_anime_indices = np.where(predictions >= 1)[0]
    recommended_anime = anime_df2.iloc[recommended_anime_indices]

    user_ratings = get_user_ratings(user_id)
    rated_anime_ids = [rating['Anime_id'] for rating in user_ratings]
    recommended_anime = recommended_anime[~recommended_anime['Anime_id'].isin(rated_anime_ids)]

    recommended_anime = recommended_anime.head(n)[['Anime_id', 'Name','English name','Score', 'Genres', 'Synopsis','Type','Episodes','Duration', 'Favorites','Scored By','Members','Image URL','Old', 'JapaneseLevel']]

    return {"recommended_anime": recommended_anime.to_dict(orient="records")}

import uvicorn
import os
if __name__ == "__main__":
    port = int(os.getenv("PORT", 4002))  # Render sẽ cung cấp cổng trong biến PORT:
    uvicorn.run("decisiontree:app", host="0.0.0.0", port=port)
