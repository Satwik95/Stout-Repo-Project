"""
Assignment for Stout.
@author: Satwik Mishra
Refrence: for feature engineering: https://www.kaggle.com/arjunjoshua/predicting-fraud-in-financial-payment-services
Dataset: https://www.kaggle.com/ntnu-testimon/paysim1
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme()


class Fraud_Analysis:
    def __init__(self, df):
        self.df = df

    def plot_transactions(self):
        """
        Plots bar graph for transaction types
        :return: None
        """
        self.df.type.value_counts().plot(kind="bar", title="Transaction Types", color=['b', 'r', 'g', 'y', 'k'])
        plt.ylabel("Number of Transactions")
        plt.xlabel("Transaction Types")
        plt.show()

    def pairplot(self, X):
        """
        Pairplot comparision of all the attributes
        :param X: processed data
        :return: None
        """
        x_fraud, x_non_fraud = X.loc[X.isFraud == 1], X.loc[X.isFraud == 0]
        subset = x_fraud + x_non_fraud.head(1000)
        subset = pd.concat([x_fraud, x_non_fraud.head(10000)], ignore_index=True)
        subset = subset.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
        sns.pairplot(subset, hue="isFraud")
        plt.show()

    def pie_chart(self, X, class_type):
        """
        Plots pie chart
        :param X: preprocessed data
        :param class_type: fraud/not fraud (1/0)
        :return: None
        """
        y = len(X.loc[(X.oldbalanceDest == 0) & (X.newbalanceDest == 0) & (X.amount != 0)]) / len(X)
        labels = ["With Balance Error", "No Balance Error"]
        x = round(y, 4) * 100
        sizes = [x, 100 - x]
        explode = (0, 0.01)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        plt.legend()
        plt.title("Balance Error vs No Error for " + class_type + " transactions")
        plt.show()

    def boxplot(self, X):
        """
        Plots boxplot for balancedDiffDest vs. isFraud attributes
        :param X: preprocessed data
        :return: None
        """
        sns.boxplot(x="isFraud", y="balanceDiffDest", data=X, showfliers=False)
        plt.title("Boxplot for Destination Account")
        plt.show()

    def correlation(self, X):
        """
        Plots the correlation matrix/heat map
        :param X: preprocessed data
        :return: None
        """
        corr = X.corr()
        sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
        plt.title("Feature Correlation Matrix")
        plt.show()

    def plot_roc(self, testY, probY):
        """
        Plots the ROC curve
        :param testY: true values
        :param probY: predicted values
        :return: None
        """
        fp, tp, _ = metrics.roc_curve(testY, probY)
        auc = metrics.roc_auc_score(testY, probY)
        plt.plot(fp, tp, label="Predictions, AUC score=" + str(auc))
        plt.legend(loc=4)
        plt.show()
        print("The ROC score for test data:", str(auc))

    def plot(self, X):
        """
        Plot helper function
        :param X: preprocessed data
        :return: None
        """
        self.plot_transactions()
        self.pairplot(X)
        x_fraud, x_non_fraud = X.loc[X.isFraud == 1], X.loc[X.isFraud == 0]
        self.pie_chart(x_fraud, "Fraud")
        self.pie_chart(x_non_fraud, "Non-Fraud")
        self.boxplot(X)
        self.correlation(X)

    def get_fraud_perc(self):
        """
        Prints the percenatge of frau transactions  in total
        :return:
        """
        x = self.df.isFraud.value_counts()[1] / self.df.isFraud.size
        y = 1 - x
        print("Percentage of the fraudulent and non fraudulent transactions", round(x, 4) * 100, round(y, 4) * 100)

    def analyse(self):
        """
        Statiscal and visal analysis/cleaning of data
        :return: None
        """
        print("General stats:-------")
        print(self.df.describe())
        print("Are there null values:", self.df.isnull().values.any())
        print("\n--------Check for imbalance-------")
        print("What are the class counts", self.df.isFraud.value_counts())
        print("------------------------------------")
        self.get_fraud_perc()

    def process_training_data(self):
        """
        Pre processing of data + feature Engineering.
        :return: X : Processed data
        """
        df = self.df
        X = df.loc[(df.type == "TRANSFER") | (df.type == "CASH_OUT")]
        X.loc[(X.oldbalanceDest == 0) & (X.newbalanceDest == 0) & (X.amount != 0), ['oldbalanceDest',
                                                                                    'newbalanceDest']] = - 1
        X['balanceDiffOrg'] = (X.newbalanceOrig - X.oldbalanceOrg) + X.amount
        X['balanceDiffDest'] = (X.oldbalanceDest - X.newbalanceDest) + X.amount
        return X

    def param_tuning(self, trainX, trainY, wts):
        """
        Performs grod search for Xgboost
        :param trainX: training features
        :param trainY: traiing labels
        :param wts: ratio of weights
        :return:None
        """
        # Grid search
        params = {'max_depth': range(4), 'gamma': (0.1, 0.2, 0), 'max_depth': range(5)}
        grid = GridSearchCV(estimator=XGBClassifier(scale_pos_weight=wts, n_jobs=4),
                            param_grid=params, scoring='roc_auc', n_jobs=4, iid=False, cv=5, verbose=1)
        grid.fit(trainX, trainY)
        print(grid.best_params_, grid.best_score_)

    def pred(self, X):
        """
        Computes the Xgboost and gradient boost predictions for given data.
        :param X: pre-processed data
        :return: None
        """
        Y = X['isFraud']
        X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud', 'isFraud'], axis=1)
        # hot-encoding of transaction type'
        X.loc[X.type == 'TRANSFER', 'type'] = 0
        X.loc[X.type == 'CASH_OUT', 'type'] = 1
        X.type = X.type.astype(int)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
        wts = sum((Y == 0)) / sum(1.0 * (Y == 1))
        # Grid search -- checking of best params
        # uncomment to compute params
        # print("Grid searching....")
        # self.param_tuning(x_train, y_train, wts)

        clf = XGBClassifier(max_depth=1, gamma=0.1, scale_pos_weight=wts, n_jobs=4)
        print("-----------------------------TRAINING XGBOOST------------------------------------")
        probs = clf.fit(x_train, y_train).predict_proba(x_test)
        probY = probs[:, 1]
        self.plot_roc(y_test, probY)

        print("-----------------------------TRAINING Gradient Boosting------------------------------------")
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        probs = clf.fit(x_train, y_train).predict(x_test)
        self.plot_roc(y_test, probs)

    def start(self):
        """
        Starting function, to call other helper functions.
        :return: None
        """
        self.analyse()
        X = self.process_training_data()
        self.plot(X)
        self.pred(X)
        print("Done!")


def main():
    df = pd.read_csv("PS_20174392719_1491204439457_log.csv")
    obj = Fraud_Analysis(df)
    obj.start()


if __name__ == "__main__":
    main()
