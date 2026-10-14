import pandas as pd

matches = pd.read_csv('matches.csv', index_col=0)

#create predictors
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype(int)
matches["day_code"] = matches["date"].dt.dayofweek

#target variable(Win or Loss)
matches["target"] = (matches["result"] == "W").astype("int")

#randomForest picks up non-linear relationships for ex. opp_code is not linear
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches["date"] < "2022-01-01"]
test = matches[matches["date"] >= "2022-01-01"]

predictors = ["venue_code", "opp_code", "hour", "day_code"]
# training using the predictors to predict the target variable
rf.fit(train[predictors], train["target"])
RandomForestClassifier(min_samples_split=10, n_estimators=50, random_state=1)
preds = rf.predict(test[predictors])

from sklearn.metrics import accuracy_score
# what percent of predictions were correct?
acc = accuracy_score(test["target"], preds)
print(acc)


# see which situations predictions were high accuracy
combined = pd.DataFrame(dict(actual=test["target"], predicted=preds))
combined = combined.reset_index(drop=True)
print(pd.crosstab(index=combined["actual"], columns=combined["predicted"]))

from sklearn.metrics import precision_score
print(precision_score(test["target"], preds))

grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Manchester City")

def rolling_averages(group, cols, new_cols):
	group = group.sort_values("date")
	# take current week out of rolling average, so closed="left"
	rolling_stats = group[cols].rolling(3, closed="left").mean()
	group[new_cols] = rolling_stats
	group = group.dropna(subset=new_cols)
	return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

print(rolling_averages(group, cols, new_cols))