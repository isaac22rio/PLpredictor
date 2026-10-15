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

# creates dataframe for each team with rolling averages for each match
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols)).reset_index()
matches_rolling.index = range(matches_rolling.shape[0]) # assign new index values from 0

def make_predictions(data, predictors):
	train = data[data["date"] < "2022-01-01"]
	test = data[data["date"] >= "2022-01-01"]
	rf.fit(train[predictors], train["target"])
	preds = rf.predict(test[predictors])
	combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
	precision = precision_score(test["target"], preds)
	return combined, precision

combined, precision = make_predictions(matches_rolling, predictors + new_cols)

print(precision)
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True) # merge combined with team, date, opponent, result columns from matches_rolling indicies are matched up based on both indexes being the same

class MissingDict(dict):
	__missing__ = lambda self, key: key

map_values = {
	"Brighton and Hove Albion": "Brighton",
	"Manchester United": "Manchester Utd",
	"Newcastle United": "Newcastle Utd",
	"Tottenham Hotspur": "Tottenham",
	"West Ham United": "West Ham",
	"Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)
combined["new_team"] = combined["team"].map(mapping)

merged = combined.merge(combined,left_on=["date", "new_team"], right_on=["date", "opponent"])
