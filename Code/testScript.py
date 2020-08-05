# -------------------------------------------------------------------
### selectors to choose which part of the test to run

test_swr = False
test_ewr = False
test_swe = False
test_json = False
test_dv = False
test_ml = True
test_stat = False

test_dv_polar = False
test_dv_scatter_all = False
test_dv_scatter_class = True

test_ml_linr = False
test_ml_logr = False
test_ml_gen_sound = True

# -------------------------------------------------------------------
### Synth wrapper test

if test_swr:
    from src import Csound_SynthWrapper as CSW

    synth = CSW.Csound_SynthWrapper()

    synth.generate_sound("non_party", [4500, 3.0])
    synth.generate_sound("party", [4500, 6.0])

# -------------------------------------------------------------------
## Extractor wrapper test

if test_ewr:
    from src import Essentia_LLD_ExtractorWrapper as EWLLD

    extractor = EWLLD.EssentiaWrapper()
    filename = "test.wav"

    features = extractor.extract_features(filename)

    print(features)

# -------------------------------------------------------------------
### Sweeper test

if test_swe:
    from src import ParameterSweeper as PW
    from src import Csound_SynthWrapper as CSW
    from src import SubSynth_SynthWrapper as SSSW
    from src import Essentia_HLD_ExtractorWrapper as EWHLD
    from src import Essentia_LLD_ExtractorWrapper as EWLLD
    from datetime import datetime

    synth = SSSW.SubSynth_SynthWrapper()
    # synth = CSW.Csound_SynthWrapper()
    extractor = EWLLD.EssentiaWrapper()

    sweeper = PW.ParameterSweeper(synth, extractor)

    startTime = datetime.now()
    sweeper.sweep_parameters()
    timeTaken = datetime.now() - startTime

    sweeper.dump_data()

    print("-------------------------------------------------------------------------------")
    print("Elapsed time: " + str(timeTaken))
    print("-------------------------------------------------------------------------------")

# -------------------------------------------------------------------
### JSON test

if test_json:
    import json

    with open("data.json") as data_json:
        data = json.load(data_json)

        print("-------------------------")

        # NOTE: this can unpack keys, use this to find name of keys
        for key in [*data["data"][0]["Parameters"]]:
            print(key + " : " + str(data["data"][0]["Parameters"][key]))
        
        print("-------------------------")

        for key in [*data["data"][0]["Features"]]:
            print(key + " : " + str(data["data"][0]["Features"][key]))

        print("-------------------------")

# -------------------------------------------------------------------
### Data visualization test

if test_dv:
    import json
    import plotly.express as pe
    import pandas as pd
    import plotly.graph_objects as pgo
    import matplotlib.pyplot as plt
    import copy

    keys = []
    values_highest = {}
    values_lowest = {}
    all_data_for_keys = {}
    data = []

    # Get all the data from json
    with open("./data/data (HLD, 2109 points).json") as data_file:
        data = json.load(data_file)

        # set up keys
        for key in [*data["data"][0]["Features"]]:
            keys.append(key)

        # Gets set of lowest and highest values for each extracted features
        for key in keys:
            all_key_val = [val["Features"][key] for val in data["data"]]
            all_data_for_keys[key] = all_key_val
            values_lowest[key] = min(all_key_val)
            values_highest[key] = max(all_key_val)

    # ----------------------------------------------------------------
    # Plot all combinations of correlations between features
    # ----------------------------------------------------------------
    
    if test_dv_scatter_all:

        all_data_copy = copy.deepcopy(all_data_for_keys)

        subplot_rows = 3
        subplot_cols = 7

        fig, axs = plt.subplots(
            subplot_rows, 
            subplot_cols, 
            sharex=True, 
            sharey=True
        )

        curr_row = 0
        curr_col = 0

        for x_key in all_data_for_keys:
            
            del all_data_copy[x_key]

            for y_key in all_data_copy:

                axs[curr_row, curr_col].scatter(
                    all_data_for_keys[x_key],
                    all_data_for_keys[y_key]
                )
                axs[curr_row, curr_col].set(
                    xlabel=x_key,
                    ylabel=y_key
                )

                if curr_col == 6: curr_row = (curr_row + 1) % subplot_rows
                curr_col = (curr_col + 1) % subplot_cols
            


        plt.show()

    # ---------------------------------------------
    # Plot it as a radar chart
    # ---------------------------------------------

    if test_dv_polar:

        fig = pgo.Figure()

        # for tup in data["data"]:
        fig.add_trace(pgo.Scatterpolar(
                r = list(values_highest.values()),
                theta =   keys,
                fill='toself'
        ))
        fig.add_trace(pgo.Scatterpolar(
                r = list(values_lowest.values()),
                theta =   keys,
                fill='toself'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                visible=True,
                range=[0, 1]
                )),
            showlegend=False
        )

        fig.show()

    # ----------------------------------------------------------------
    # Plot scatter graphs of classifications on parameter axis
    # ----------------------------------------------------------------

    if test_dv_scatter_class:

        data_list = data["data"]

        # for point in data_list:
        #         for feature in point["Features"]:
        #             point["Features"].update({feature : round(point["Features"][feature])})

        key = "party"

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # points = [list(p["Parameters"].values()) for p in data_list if p["Features"][key] == 1]
        points = [list(p["Parameters"].values()) for p in data_list]
        points_x = [p[0] for p in points]
        points_y = [p[1] for p in points]
        points_z = [p["Features"][key] for p in data_list]

        ax.scatter(points_x, points_y, points_z)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("LFO freq (Hz)")
        ax.set_zlabel(key + " probability")

        plt.show()

# -------------------------------------------------------------------
### Machine Learning test

import json
from sklearn import linear_model as lm
from sklearn.svm import SVR, SVC, LinearSVR
from sklearn.metrics import mean_squared_error as mse, r2_score as r2s
from sklearn.multioutput import MultiOutputRegressor as MOR, MultiOutputClassifier as MOC
from datetime import datetime
import random

if test_ml:

    data_list = []

    ###################################
    # Open the necessary data files here and add them to the data_list

    # with open("./data/data (SubSynth, 4 params 710 points) (2 osc types, no PWM change).json") as data_file:
    with open("./data/data (HLD, 2109 points).json") as data_file:
        data = json.load(data_file)

        data_list = data["data"]

    with open("./data/data (SubSynth, 4 params, 1420 points) (sq wave osc, with PWM change).json") as data_1_file:
        data = json.load(data_1_file)

        data_list = data_list + data["data"]

    with open("./data/data (SubSynth, 7 params, 17040 points) (sq, 2osc).json") as data_1_file:
        data = json.load(data_1_file)

        data_list = data_list + data["data"]
    
    with open("./data/data (SubSynth, 7 params, 34080 points) (sq, sq).json") as data_1_file:
        data = json.load(data_1_file)

        data_list = data_list + data["data"]

    ###################################

    # Linear regression
    # Or Support Vector Regression
    if test_ml_linr:

        data_list_test = data_list[0::5]
        data_list_train = [x for x in data_list if x not in data_list_test]

        total_acc_cumu = 0
        total_acc_count = 0

        with open("./data/test.txt", 'w') as pred_file:

            param_list_test = [list(p["Parameters"].values()) for p in data_list_test]
            feat_list_test = [list(p["Features"].values()) for p in data_list_test]

            param_list_train = [list(p["Parameters"].values()) for p in data_list_train]
            feat_list_train = [list(p["Features"].values()) for p in data_list_train]

            # regr = lm.LinearRegression()

            # multi output regressor to analyse all synth parameters at once
            estimator = SVR()
            regr = MOR(estimator, -1)

            # set x to extracted features, and y as parameters
            # can then predict parameters (y) based on features (x)

            startTime = datetime.now()
            print("Fitting")
            regr.fit(feat_list_train, param_list_train)

            print("Predicting")
            param_pred = regr.predict(feat_list_test)
            timeTaken = datetime.now() - startTime

            print("-------------------------------------------------------------------------------")
            print("Elapsed time: " + str(timeTaken))
            print("-------------------------------------------------------------------------------")

            # Compute accuracy value
            lower_bound = 0.9
            upper_bound = 1.1
            count_correct = 0
            for (a_1, a_2) , (p_1, p_2) in zip(param_list_test, param_pred):
                if  a_1 <= p_1 * upper_bound and \
                    a_1 >= p_1 * lower_bound and \
                    a_2 <= p_2 * upper_bound and \
                    a_2 >= p_2 * lower_bound \
                    : count_correct = count_correct + 1
            accuracy = (count_correct / len(param_list_test)) * 100

            total_acc_count = total_acc_count + 1
            total_acc_cumu = total_acc_cumu + accuracy

            pred_file.write("Accuracy: " + str(accuracy) + "\n")
            pred_file.write("Coefficient of determination: " + str(r2s(param_list_test, param_pred)) + "\n")
            pred_file.write("Mean squared error: " + str(mse(param_list_test, param_pred)) + "\n")
            pred_file.write("--------------------------------\n\n")

    # Logistic regression
    # Or support vector classification
    if test_ml_logr:
        
        for point in data_list:
            for feature in point["Features"]:
                point["Features"].update({feature : round(point["Features"][feature])})
        
        data_list_test = data_list[0::5]
        data_list_train = [x for x in data_list if x not in data_list_test]

        total_acc_cumu = 0
        total_acc_count = 0

        with open("./data/prediction_data_all_HLDs_LR (2109 data points) (with confusion matrix).txt", 'w') as pred_file:

            for key in data_list[0]["Features"].keys():

                param_list_test = [list(p["Parameters"].values()) for p in data_list_test]
                feat_list_test = [p["Features"][key] for p in data_list_test]

                param_list_train = [list(p["Parameters"].values()) for p in data_list_train]
                feat_list_train = [p["Features"][key] for p in data_list_train]

                if all(i == 0 for i in feat_list_train) or all(i == 1 for i in feat_list_train):
                    pred_file.write(key + " has been skipped\n")
                    pred_file.write("--------------------------------\n\n")
                    continue

                # regr = lm.LogisticRegression()
                regr = SVC()
                regr.fit(param_list_train, feat_list_train)

                feat_pred = regr.predict(param_list_test)

                true_pos = 0
                true_neg = 0
                false_pos = 0
                false_neg = 0
                for a, p in zip(feat_list_test, feat_pred):
                    if p == 1:
                        if a == p: true_pos = true_pos + 1
                        else: false_pos = false_pos + 1
                    else:
                        if a == p: true_neg == true_neg + 1
                        else: false_neg == false_neg + 1

                accuracy = (true_pos + true_neg)/(true_pos + true_neg + false_pos + false_neg) * 100
                key_pos_precision = (true_pos / (true_pos + false_pos)) * 100 if true_pos + false_pos != 0 else 0
                key_neg_precision = (true_neg / (true_neg + false_neg)) * 100 if true_neg + false_neg != 0 else 0
                key_pos_recall = (true_pos / (true_pos + false_neg)) * 100 if true_pos + false_neg != 0 else 0
                key_neg_recall = (true_neg / (true_neg + false_pos)) * 100 if true_neg + false_pos != 0 else 0

                total_acc_count = total_acc_count + 1
                total_acc_cumu = total_acc_cumu + accuracy

                pred_file.write(key + "\n")
                pred_file.write("Accuracy: " + str(accuracy) + "\n")
                pred_file.write("--------------------------------\n")
                pred_file.write("True positive: " + str(true_pos) + "\n")
                pred_file.write("False positive: " + str(false_pos) + "\n")
                pred_file.write("True negative: " + str(true_neg) + "\n")
                pred_file.write("False negative: " + str(false_neg) + "\n")
                pred_file.write("--------------------------------\n")
                pred_file.write("Positive precision: " + str(key_pos_precision) + "\n")
                pred_file.write("Negative precision: " + str(key_neg_precision) + "\n")
                pred_file.write("Positive recall: " + str(key_pos_recall) + "\n")
                pred_file.write("Negative recall: " + str(key_neg_recall) + "\n")
                pred_file.write("Mean squared error: " + str(mse(feat_list_test, feat_pred)) + "\n")
                pred_file.write("--------------------------------\n\n")

            total_acc = total_acc_cumu / total_acc_count

            pred_file.write("Overall accuracy: " + str(total_acc) + "\n")

    # generate a sound based on a trained linear regressor
    # on the LLD features of the simple synthesizer
    if test_ml_gen_sound:

        keys = []
        values_highest = {}
        values_lowest = {}
        all_data_for_keys = {}
        data = []

        # Get all the data from json
        with open("./data/data (LLD, 2109 points).json") as data_file:
            data = json.load(data_file)

            # set up keys
            for key in [*data["data"][0]["Features"]]:
                keys.append(key)

            # Gets set of lowest and highest values for each extracted features
            for key in keys:
                all_key_val = [val["Features"][key] for val in data["data"]]
                all_data_for_keys[key] = all_key_val
                values_lowest[key] = min(all_key_val)
                values_highest[key] = max(all_key_val)

        data_list_test = data["data"][0::5]
        data_list_train = [x for x in data["data"] if x not in data_list_test]

        param_list_train = [list(p["Parameters"].values()) for p in data["data"]]
        feat_list_train = [list(p["Features"].values()) for p in data["data"]]

        param_list_test = [list(p["Parameters"].values()) for p in data_list_test]
        feat_list_test = [list(p["Features"].values()) for p in data_list_test]

        regr = lm.LinearRegression()
        regr.fit(feat_list_train, param_list_train)

        random.seed(0)

        # for each key in the features, generate a random value
        count = 0
        for _ in range(100):
            features = []

            for key in keys:
                rand = random.random()
                feat = values_lowest[key] + (rand * (values_highest[key] - values_lowest[key]))
                features.append(feat)

            # print(features)
            param_pred = regr.predict([features])
            print(param_pred)
            if param_pred[0][0] > 27.5 and param_pred[0][0] < 7040.0 and param_pred[0][1] > 2.5 and param_pred[0][1] < 7.0:
                count = count + 1
        
        print(count)

# -------------------------------------------------------------------
### Statistical tests

import json
import numpy as np
from statsmodels.multivariate.manova import MANOVA

if test_stat:

    data_list = []

    with open("./data/data (SubSynth, 2 param, 355 points).json") as data_file:
        data = json.load(data_file)
        data_list = data["data"]

    # for point in data_list:
    #     del point["Features"]["acoustic"]
    #     for feature in point["Features"]:
    #         point["Features"].update({feature : round(point["Features"][feature])})

    # var_indep = np.asarray([list(x["Features"].values()) for x in data_list])
    # var_dep = np.asarray([list(x["Parameters"].values()) for x in data_list])

    # # endog - DVs, exog - IVs
    # mnv = MANOVA(var_dep, var_indep)

    # with open("./data/manova_data (HLD, 2109 points).txt", "w") as res_file:
    #     res_file.write(str(mnv.mv_test()))

    x_1 = [p["Features"]["loudness_ebu128_momentary"] for p in data_list]
    x_2 = [p["Features"]["loudness_ebu128_short_term"] for p in data_list]
    y = [p["Parameters"]["o1 amplitude"] for p in data_list]

    print("momentary")
    r = np.corrcoef(x_1, y)
    print(r)
    print("short_term")
    r = np.corrcoef(x_2, y)
    print(r)