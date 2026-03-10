# Developed by Pedro Ribeiro / Pedro M. Rodrigues
# Core libraries
import time
import pandas as pd # toolbox for working with DataFrames
import numpy as np # toolbox for working with numerical arrays

# ML models and techniques
from sklearn.svm import SVC # Support Vector Machine
from sklearn.feature_selection import SelectKBest # feature selector
from sklearn.feature_selection import chi2 # chi2 feature selection algorithm
from sklearn.decomposition import FastICA # feature combination/reduction via ICA

# General settings
import warnings
warnings.filterwarnings('ignore') # suppress warnings in output
from tqdm.notebook import tqdm # progress bars

def normalized_data(df, t):
    """
    Normalize a DataFrame based on the selected method:
    t == 1 -> Min-Max normalization
    t == 2 -> Mean normalization
    otherwise -> Standardization
    """
    if (t == 1):
        d = df.copy() # copy for min-max normalization
        for each_collum in range(0, df.shape[1]):
            max = df.iloc[:, each_collum].max()
            min = df.iloc[:, each_collum].min()
            d.iloc[:, each_collum] = (d.iloc[:, each_collum] - min) / (max - min)
    elif (t == 2):
        d = df.copy() # copy for mean normalization
        for each_collum in range(0, df.shape[1]):
            max = df.iloc[:, each_collum].max()
            min = df.iloc[:, each_collum].min()
            mean = df.iloc[:, each_collum].mean()
            d.iloc[:, each_collum] = (d.iloc[:, each_collum] - mean) / (max - min)
    else:
        d = df.copy() # copy for standardization
        for each_collum in range(0, df.shape[1]):
            mean = df.iloc[:, each_collum].mean()
            std = df.iloc[:, each_collum].std()
            d.iloc[:, each_collum] = (d.iloc[:, each_collum] - mean) / (std)

    return d

# Load feature selection utilities
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import GenericUnivariateSelect

def feature_selector(X_train, y_train, type, i):
    """
    Select i features based on algorithm chosen by 'type':
    1 -> f_classif
    2 -> mutual_info_classif
    3 -> chi2
    4 -> SelectFdr (kept as in original code)
    5 -> SelectFwe (kept as in original code)
    """
    if (type == 1):
        # ANOVA F-value for classification tasks
        bestfeatures = SelectKBest(score_func=f_classif, k=i)
    elif (type == 2):
        # Mutual information for discrete target
        bestfeatures = SelectKBest(score_func=mutual_info_classif, k=i)
    elif (type == 3):
        # Chi-squared for non-negative features in classification
        bestfeatures = SelectKBest(score_func=chi2, k=i)
    elif (type == 4):
        # Selection based on false discovery rate (as in original version)
        bestfeatures = SelectKBest(score_func=SelectFdr, k=i)
    elif (type == 5):
        # Selection based on family-wise error rate (as in original version)
        bestfeatures = SelectKBest(score_func=SelectFwe, k=i)

    # Fit selector and get best feature indices
    fit = bestfeatures.fit(X_train, y_train)
    cols_idxs = fit.get_support(indices=True)

    # Extract only selected training features
    Xt = X_train.iloc[:, cols_idxs]
    return Xt, cols_idxs

# 3rd step - load and define classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF

# List of models used in benchmarking
classifiers = [
    SVC(gamma='auto', probability=True),
    KNeighborsClassifier(),
    LogisticRegression(solver='lbfgs'),
    # BaggingClassifier(),
    # GaussianNB(),
    DecisionTreeClassifier(max_depth=5),
    AdaBoostClassifier(),
    # LinearDiscriminantAnalysis(),
    # QuadraticDiscriminantAnalysis(),
    LinearSVC(),
    # OneVsRestClassifier(LinearSVC(random_state=0)),
]

# Metrics and validation
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix
import matplotlib.patches as patches
from sklearn.metrics import roc_curve, auc


def holdOut(model, train, trainTarget, test, testTarget):
    """
    Train a hold-out model, compute metrics on train/test,
    and return accuracy, precision, recall, F1, AUC, confusion matrix, and ROC curve.
    """

    # Helper structures (kept as in original version)
    y_true_all, y_pred_all = [], []
    train_accuracies, test_accuracies = [], []
    train_precision, test_precision = [], []
    train_recall, test_recall = [], []
    train_f1, test_f1 = [], []
    areaUCTrain, areaUCTest = [], []

    # Train model
    estimator = model.fit(train, trainTarget)

    # Predictions on train and test
    y_train_pred = estimator.predict(train)
    y_test_pred = estimator.predict(test)

    # ROC curves (train and test)
    fprTrain, tprTrain, thresholds = roc_curve(trainTarget, y_train_pred, pos_label=trainTarget.max())
    fprTest, tprTest, thresholds = roc_curve(testTarget, y_test_pred, pos_label=testTarget.max())

    # AUC (train and test)
    train_Auc = auc(fprTrain, tprTrain)
    test_Auc = auc(fprTest, tprTest)

    # Train metrics
    train_accuracies = accuracy_score(trainTarget, y_train_pred)
    train_precision = precision_score(trainTarget, y_train_pred, average='macro')
    train_recall = recall_score(trainTarget, y_train_pred, average='macro')
    train_f1 = (2 * train_precision * train_recall) / (train_precision + train_recall)

    # Test metrics
    test_accuracies = accuracy_score(testTarget, y_test_pred)
    test_precision = precision_score(testTarget, y_test_pred, average='macro')
    test_recall = recall_score(testTarget, y_test_pred, average='macro')
    test_f1 = (2 * test_precision * test_recall) / (test_precision + test_recall)

    # Convert to percentage
    train_accuracies = train_accuracies * 100
    test_accuracies = test_accuracies * 100
    train_precision = train_precision * 100
    test_precision = test_precision * 100
    train_recall = train_recall * 100
    test_recall = test_recall * 100
    train_f1 = train_f1 * 100
    test_f1 = test_f1 * 100

    # Final confusion matrix (test)
    final_conf_matrix = confusion_matrix(testTarget, y_test_pred)

    return train_accuracies, train_precision, train_recall, train_f1, test_accuracies, test_precision, test_recall, test_f1, final_conf_matrix, train_Auc, test_Auc, fprTest, tprTest

# Base path for Excel file
path = 'database Path' # folder containing the XLSX file

# Spectral region dictionary (bins per region)
regIndex = {
    'R1': [3000, 2800],
    'R2': [1750, 1500],
    'R3': [1500, 1200],
    'R4': [1200, 900],
    'AllBands': [3000, 900]
}

# Region list for iteration
cregIndexKeys = list(regIndex.keys())

# Excel sheet names
excelSheets = pd.ExcelFile(path + "spectra 2.xlsx").sheet_names

# Performance results DataFrame
perf_results = pd.DataFrame()
o = 0 # incremental index for result columns

# Read training/testing data (sheet 0 and sheet 1)
dfTrain = pd.read_excel(path + 'spectra 2.xlsx', sheet_name=excelSheets[0])
dfTest = pd.read_excel(path + 'spectra 2.xlsx', sheet_name=excelSheets[1])

# Target split
trainTarget = dfTrain.pop('Target')
testTarget = dfTest.pop('Target')

# Main loop by region
for region in tqdm(cregIndexKeys, desc='Color', leave=0):
    # Local copies to avoid changing base DataFrames
    xTrain = dfTrain.copy()
    xTest = dfTest.copy()

    # Select spectral interval for current region
    xTrain = xTrain.iloc[:, xTrain.columns.get_loc(regIndex[region][0]):xTrain.columns.get_loc(regIndex[region][1]) + 1]
    xTest = xTest.iloc[:, xTest.columns.get_loc(regIndex[region][0]):xTest.columns.get_loc(regIndex[region][1]) + 1]

    # AllBands case: remove specific bins between R1 and R2
    if region == 'AllBands':
        cols_to_drop = [c for c in xTrain.columns if 1752 <= int(c) <= 2798]
        xTrain = xTrain.drop(columns=cols_to_drop)
        xTest = xTest.drop(columns=cols_to_drop)

    # Active data for processing (kept as in original)
    d_n = xTrain

    # Output helper DataFrames
    confMat = pd.DataFrame() # confusion matrices
    rocAUC = pd.DataFrame() # ROC points
    featuresSelectedDF = pd.DataFrame() # selected features

    # Feature combination loop (ICA components)
    for combValue in tqdm(range(1, 31, 1), desc='CombinationVal', leave=0):
        combData = FastICA(n_components=combValue)

        # Feature selection loop
        for j in tqdm(range(30, d_n.shape[1], 10), desc='Feature Selection', leave=0):
            # Feature selection timing
            startFS = time.time()
            df_nf, cols_idxs = feature_selector(d_n, trainTarget, 1, j)
            Xtest = xTest.iloc[:, cols_idxs]
            endFS = time.time()
            timeFS = endFS - startFS

            # Combination timing (ICA)
            startComb = time.time()
            df_nf = combData.fit_transform(df_nf)
            Xtest = combData.transform(Xtest)
            endComb = time.time()
            timeComb = endComb - startComb

            # Store selected feature names
            featuresSelected = list(d_n.columns[cols_idxs])

            # NaN padding to align exported column sizes
            featsSelectedName = np.ones(d_n.shape[1] - len(cols_idxs))
            featsSelectedName[:] = np.nan
            featuresSelected.extend(list(featsSelectedName))

            # Selected feature column for current configuration
            featuresSelectedDF[
                combData.__class__.__name__ + 'Number_' + str(combValue) + '_FeatureSelectionNumber_' + str(j)
            ] = featuresSelected

            # Test all configured classifiers
            for classifier in classifiers:
                try:
                    # Training + inference timing
                    startClassifier = time.time()
                    trainAcc, trainPrec, trainRec, trainF1, testAcc, testPrec, testRec, testF1, conf_matrix, train_Auc, test_Auc, fpr, tpr = holdOut(
                        classifier, df_nf, trainTarget, Xtest, testTarget
                    )
                    endClassifier = time.time()
                    timeClassifier = endClassifier - startClassifier

                    # Flatten confusion matrix for column storage
                    conf_matrix = conf_matrix.reshape(-1)

                    # FPR length normalization for export
                    fpr = list(fpr)
                    fprLen = np.ones(50 - len(fpr))
                    fprLen[:] = np.nan
                    fpr.extend(list(fprLen))

                    # TPR length normalization for export
                    tpr = list(tpr)
                    tprLen = np.ones(50 - len(tpr))
                    tprLen[:] = np.nan
                    tpr.extend(list(tprLen))

                    # Total runtime for current configuration
                    timeCount = (timeFS + timeComb + timeClassifier)

                    # Store confusion matrix
                    confMat[
                        combData.__class__.__name__ + 'Number_' + str(combValue) + '_FeatureSelectionNumber_' + str(j) + '_Classifier_' + classifier.__class__.__name__
                    ] = conf_matrix

                    # Store ROC (FPR/TPR)
                    rocAUC[
                        'FPR_' + combData.__class__.__name__ + 'Number_' + str(combValue) + '_FeatureSelectionNumber_' + str(j) + '_Classifier_' + classifier.__class__.__name__
                    ] = fpr
                    rocAUC[
                        'TPR_' + combData.__class__.__name__ + 'Number_' + str(combValue) + '_FeatureSelectionNumber_' + str(j) + '_Classifier_' + classifier.__class__.__name__
                    ] = tpr

                    # Store performance metrics for current configuration
                    perf_results[o] = [
                        region, combData.__class__.__name__, combValue, j, classifier.__class__.__name__,
                        trainAcc, trainPrec, trainRec, trainF1, train_Auc,
                        testAcc, testPrec, testRec, testF1, test_Auc, timeCount
                    ]
                    o = o + 1

                except Exception as e:
                    # If one classifier/configuration fails, skip and continue
                    pass

        # Export partial results by region/combination
        confMat.to_csv(path + 'Results' + '/ConfMat_' + combData.__class__.__name__ + '_Color_' + region + '_.csv')
        rocAUC.to_csv(path + 'Results' + '/rocAUC_' + combData.__class__.__name__ + '_Color_' + region + '_.csv')
        featuresSelectedDF.to_csv(path + 'Results' + '/Features_' + combData.__class__.__name__ + '_Color_' + region + '_.csv')

# Final performance table post-processing
perf_results = perf_results.T
perf_results.columns = [
    'Color', 'Combination', 'Combination_number_feat', '# features', 'classifier',
    "Mean Training Accuracy", "Mean Training Precision", "Mean Training Recall", "Mean Training F1 Score", "Mean Training AUC",
    "Mean Validation Accuracy", "Mean Validation Precision", "Mean Validation Recall", "Mean Validation F1 Score", "Mean Validation AUC",
    'ExecTime'
]

# Final export to Excel
perf_results.to_excel(path + 'ResultsP' + '/resultados.xlsx')
