import luigi
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
import ast

class CleanDataTask(luigi.Task):
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='clean_data.csv')


    def run(self):
        df = pd.read_csv("airline_tweets.csv", encoding="iso-8859-1", usecols=['airline_sentiment', 'tweet_coord'])
        df = df[df['tweet_coord'].notna()]
        df.drop(df[df.tweet_coord == '[0.0, 0.0]'].index, inplace=True)
        df.to_csv(self.output_file, index=False, encoding="iso-8859-1")

    def output(self):
        return luigi.LocalTarget(self.output_file)


class TrainingDataTask(luigi.Task):
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')

    def requires(self):
        return CleanDataTask(self.tweet_file)

    def run(self):
        cities_df = pd.read_csv(self.cities_file, encoding="iso-8859-1")
        df = pd.read_csv(self.input().path, encoding="iso-8859-1")
        df["tweet_coord"] = df["tweet_coord"].str.findall(r"[-+]?\d*\.\d+|\d+")

        def closest_city_finder(x, minimum=float("inf")):
            min_city_name = None
            x_lat, x_long = float(x[0]), float(x[1])
            for lat, longtitude, name in zip(cities_df['latitude'], cities_df['longitude'], cities_df['name']):
                dist = (x_lat - lat) ** 2 + (x_long - longtitude) ** 2
                if dist < minimum:
                    min_city_name = name
                    minimum = dist
            return min_city_name

        # find the closest city name using Euclidean distance
        df["closest_city_name"] = [closest_city_finder(x) for x in df['tweet_coord']]

        # convert city names to one hot vectors
        name_onehot = pd.get_dummies(cities_df['name'], columns=['name'])

        # create a mapping between closest city name and one hot vector
        dict_ = {}
        for k, v in zip(cities_df['name'], name_onehot.values.tolist()):
            if k not in dict_:
                dict_[k] = v

        # add column X that is the one hot vector for each string.
        df["X"] = df["closest_city_name"].map(lambda x: list(dict_[x]))

        # convert sentiment to one hot values as given in the problem statement.
        dict_ = {'negative': 0, 'neutral': 1, 'positive': 2}
        df['y'] = df["airline_sentiment"].map(lambda x: dict_[x])

        output_df = pd.DataFrame(df[["X", "y", "closest_city_name"]], columns=["X", "y", "closest_city_name"])
        output_df.to_csv(self.output_file, encoding="ISO-8859-1", index=False)

    def output(self):
        return luigi.LocalTarget(self.output_file)



class TrainModelTask(luigi.Task):

    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')

    def requires(self):
        return TrainingDataTask(self.tweet_file)

    def run(self):
        features = pd.read_csv(self.input().path, encoding="ISO-8859-1")
        # convert string of lists to lists.
        X = [ast.literal_eval(x) for x in features["X"].values.tolist()]
        y = features['y'].values.tolist()
        X = np.asarray(X).astype(float)

        # basic classification model, since the problem statements clearly states to focus only on the pipeline.
        clf = LogisticRegression(random_state=7, multi_class='multinomial').fit(X, y)
        joblib.dump(clf, 'model.pkl')

    def output(self):
        return luigi.LocalTarget(self.output_file)


class ScoreTask(luigi.Task):
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')

    def requires(self):
        return {"model": TrainModelTask(self.tweet_file), "features": TrainingDataTask(self.tweet_file)}

    def run(self):
        features = pd.read_csv(self.input()["features"].path, encoding="ISO-8859-1")
        X = [ast.literal_eval(x) for x in features["X"].values.tolist()]
        y = features['y'].values.tolist()
        X = np.asarray(X).astype(float)

        # load model and make predictions
        clf_load = joblib.load(self.input()["model"].path)
        y_pred_prob = clf_load.predict_proba(X)

        score = pd.DataFrame(y_pred_prob, columns=["negative", "neutral", "positive"])
        score["PREDICTED_LABEL"] = [np.argmax(x) for x in score[["negative", "neutral", "positive"]].values.tolist()]
        score["TRUE_LABEL"] = y
        score["city"] = features["closest_city_name"]

        score = score.drop_duplicates()
        score = score.sort_values("positive", ascending=False)

        score.to_csv(self.output_file, encoding="ISO-8859-1", index=False)

    def output(self):
        return luigi.LocalTarget(self.output_file)

if __name__ == "__main__":
    luigi.run()
