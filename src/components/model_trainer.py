import os
import sys
from dataclasses import dataclass
sys.path.insert(0, 'src')

#Importing libraries for models
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.tree import export_graphviz
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from exception import CustomException
from logger import logging

from utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
       
            target_column_name="class"

            X_train=train_array.drop(columns=[target_column_name],axis=1)
            y_train=train_array[target_column_name]

            X_test=test_array.drop(columns=[target_column_name],axis=1)
            y_test=test_array[target_column_name]
            
            logging.info("Defining models and their parameters")
            models = {
                "Logistic Regression": LogisticRegression(),
                "Support Vector Classifier": SVC(),
                "Decision Tree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier()
            }
            params={
                "Logistic Regression": {
                 #   'penalty':['none','l1','l2','elasticnet']                
                },
                "Support Vector Classifier":{
                #    'kernel':['poly', 'rbf', 'sigmoid']
                },
                "Decision Tree": {
                #    'criterion':['gini', 'entropy', 'log_loss'],
                #     'splitter':['best','random'],
                 #    'max_features':['sqrt','log2']
                },
                "KNN":{
                    'n_neighbors': range(1, 21, 2),
                 #   'weights': ['uniform', 'distance'],
                 #   'metric': ['euclidean', 'manhattan', 'minkowski']
                },
                "Naive Bayes":{
                
                },
                "Random Forest":{
                  #  'criterion':['gini', 'entropy', 'log_loss'],
                  #  'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            acc_score = accuracy_score(y_test, predicted)
            logging.info("Returning accuracy score of model")
            return acc_score
            
 
        except Exception as e:
            raise CustomException(e,sys)