import pandas as pd
import numpy as np
import random
import importlib
import configparser

np.seterr(divide='ignore')

def cutoff_truths(actual, proba, cutoff):
    predicted = (proba > cutoff)
    truths = actual + predicted * 2

    truths = (truths.replace(3, 'TP')
              .replace(2, 'FP')
              .replace(1, 'FN')
              .replace(0, 'TN'))
    return truths

def map_cutoff_ranges(actual, proba, thresholds, baseline = False):
    range = pd.Series(np.arange(0.1,0.9,0.01))
    if baseline == False:
        truth_matrix = range.apply(lambda x: cutoff_truths(actual,proba, x)).transpose()
        truth_matrix.columns = np.arange(0.1,0.9,0.01)
    else:
        truth_matrix = range.apply(lambda x: cutoff_truths(actual, proba, thresholds)).transpose()
        truth_matrix.columns = np.arange(0.1,0.9,0.01)

    truth_summary = pd.DataFrame()
    TP = (truth_matrix == 'TP').sum( axis = 0)
    TN = (truth_matrix == 'TN').sum( axis = 0)
    FP = (truth_matrix == 'FP').sum( axis = 0)
    FN = (truth_matrix == 'FN').sum( axis = 0)

    truth_summary = pd.DataFrame()
    truth_summary['STP'] = (TP + FP) / (TP + FP + TN + FN)
    truth_summary['ACC'] = (TP + TN) / (TP + TN + FP + FN)
    truth_summary['TPR'] = (TP) / (TP + FN)
    truth_summary['PPV'] = (TP) / (TP + FP)
    truth_summary['FPR'] = (FP) / (FP + TN)
    
    return truth_summary

def map_ratios(actual, proba, threshold, demo, advantaged_label):
    adv_actual = actual[demo == advantaged_label]
    adv_proba = proba[demo == advantaged_label]
    adv_threshold = threshold[demo == advantaged_label]
    
    adv_tm = map_cutoff_ranges(adv_actual, adv_proba, adv_threshold, baseline = True)

    tm_dict = {}
    for d in demo.unique():
        if d == advantaged_label:
            continue
            #metrics = map_cutoff_ranges(actual[demo == d], proba[demo == d], threshold[demo == d], baseline = True)
        else:
            metrics = map_cutoff_ranges(actual[demo == d], proba[demo == d], threshold[demo == d])
        #print(metrics)
        #ratios = abs(np.log(metrics / adv_tm))
        #ratios = metrics
        
        ratios = (metrics / adv_tm)
        ratios[ratios > 1] = 1
        ratios = abs(np.log(ratios))

        ratios.columns = metrics.columns + '_loss'
        combined = pd.concat([metrics, ratios], axis = 1)
        combined = combined.reset_index()
        combined['demo_label'] = d
        combined = combined.rename(columns = {'index':'threshold'})
        tm_dict[d] = combined

    output = pd.concat(tm_dict.values()).replace([np.inf, -np.inf], np.nan).dropna()
    return output

def optimize_thresholds(actual, proba, threshold, demo, advantaged_label):
    ratios = map_ratios(actual, proba, threshold,  demo, advantaged_label)
    #ratios['normalized_score'] = abs(np.log(ratios[[col for col in ratios.columns if not (col.endswith('_loss') or col in('threshold', 'demo_label'))]])).apply(lambda x: (x - x.mean())/x.std(), axis = 0).sum(axis = 1)
    ratios['normalized_loss_score'] = ratios[[col for col in ratios.columns if col.endswith('_loss')]].apply(lambda x: (x - x.mean())/x.std(), axis = 0).sum(axis = 1)
    #ratios['combined_normalized_score'] =  ratios['normalized_score'] + ratios['normalized_loss_score']
    top = ratios.sort_values(by= ['demo_label', 'normalized_loss_score'], ascending = True).groupby('demo_label').head(1)[['demo_label','threshold']]

    output = pd.DataFrame()
    output['demo_label'] = demo

    top = pd.concat([top, pd.DataFrame({'demo_label':[advantaged_label], 'threshold': [0.5]})])
    output = output.merge(top, on = 'demo_label', how = 'left').threshold.values

    return output

def return_optimized_thresholds(actual, proba, demo_data, demo_dict, reps = 30, thresholds_only = False):
    i = 0
    threshold = np.repeat(0.5, len(proba))
    tracking_df = pd.DataFrame()
    tracking_df[i] = threshold
    choice = random.choice(list(demo_dict.keys()))
    
    for n in range(1,reps+1):
        print("Debiasing iteration " + str(n)+"/"+str(reps))
        demo = choice 
        results = optimize_thresholds(actual, proba, threshold, demo_data[demo], demo_dict[demo])
        threshold = (threshold + results) / 2
        i += 1
        tracking_df[i] = threshold
        new_demo = demo_dict.copy()
        new_demo.pop(choice)

        choice = random.choice(list(new_demo.keys()))
    
    output = demo_data.copy()
    output['threshold'] = tracking_df.iloc[:,len(tracking_df.columns)-5:len(tracking_df.columns)].mean(axis = 1)

    if thresholds_only == False:
        output = output.drop_duplicates()
    else:
        output = output['threshold']
        
    return output

def model_agnostic_adjustment(modeling_dataset, y, proba, D, demo_dict, reps = 30):
    thresholds = return_optimized_thresholds(y, proba, D, demo_dict, reps, thresholds_only = True)
    output = modeling_dataset.copy()
    output['debiased_prediction'] = 0 
    output.loc[proba > thresholds, 'debiased_prediction'] = 1
    
    return output

class debiased_model:
    debiased_thresholds = pd.DataFrame()

    def __init__(self, config):
        self.model_settings = dict(config['model settings'])
        self.model_arguments = dict(config['model arguments'])
        for key, value in self.model_arguments.items():
            try:
                self.model_arguments[key] = int(value)
            except Exception:
                try:
                    self.model_arguments[key] = float(value)
                except Exception:
                    pass

        self.dataset_info = dict(config['dataset information'])

        self.model_module = importlib.import_module(self.model_settings['model_module'])

        model_class = getattr(self.model_module, self.model_settings['model_class'])

        self.model = model_class(**self.model_arguments)

        self.predict_proba = getattr(self.model, self.model_settings['predict_proba_method'])

    def fit(self, X, y, D, demo_dict, reps = 30, *args, **kwargs):
        fit_method = getattr(self.model, self.model_settings['fit_method'])
        fit_method(X, y, *args, **kwargs)

        proba_results = self.model.predict_proba(X)[:,1]

        self.debiased_thresholds = return_optimized_thresholds(y, proba_results, D, demo_dict, reps)

    def predict(self, X, D, *args, **kwargs):
        proba_values = self.predict_proba(X)[:,1]
        probabilities = pd.DataFrame({"proba": proba_values})

        prediction_df = pd.concat([D, probabilities], axis = 1)
        prediction_df = prediction_df.merge(self.debiased_thresholds, on = list(D.columns), how = 'left')
        prediction_df['predicted'] = 0
        prediction_df.loc[prediction_df['proba'] > prediction_df['threshold'], 'predicted'] = 1

        return prediction_df[['predicted']]
    
    def get_debiased_thresholds(self):
        return self.debiased_thresholds