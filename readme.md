#######################
##Bias Measurement Tool
#######################
CLI tool usage: measure_disparity.py [-h] settings.ini 

Settings	:	settings.ini file 
-h, --help	:	show this help message and exit 

The settings ini file for must include [dataset information], [paths], and [other].

Sample CLI usages: 
python measure_disparity.py settings-measure.ini
python measure_disparity.py -h

Sample settings ini file:
[dataset information]
target_variable: actual
demo_variables: RACE,GENDER,INCOME
advantaged_groups: white,M,High
probability_column: proba
predicted_column: predicted
weight_column: n/a

[paths]
input_path: ~/data/model_results.csv

[other]
display_metrics: STP,TPR,PPV,FPR,ACC

Besides the CLI the tool can also be more directly used and customized by importing measure_disparity_methods. The main
functions of measure_disparity_methods are described below.

Function to intialize a measured object from measure_disparity_methods.py
measure_disparity_methods.measured(dataset,
        democols,
        intercols,
        actualcol='actual',
        predictedcol='predicted',
        probabilitycol='probability',
        weightscol='weights')
    dataset <class 'pandas.core.frame.DataFrame'>: data generated from a model you which to analyze
    democols <class 'list'>: list of strings of demographic column names
    intercols <class 'list'>: list of lists with pairs of strings of demo column names you wish to see interactions of
    actualcol <class 'str'>: string of column name with actual values formatted as 0 or 1
    predictedcol <class 'str'>: string of column name with predicted values formatted as 0 or 1
    probabilitycol <class 'str'>: string of column name with probability values formatted as floats 0 to 1
    weightscol <class 'str'>: string of column name with weights formatted as ints or floats
    Returns
    measured <class 'measure_disparity.measured'>: object of the class from measure_disparity.py

Function to draw a plot of any number of given metrics for a given demographic column
measured.MetricPlots(colname, 
            privileged, 
            draw=True, 
            metrics=['STP', 'TPR', 'PPV', 'FPR', 'ACC'], 
            graphpath=None)
    colname <class 'str'>: string of demographic column name
    privileged <class 'str'>: string of name for the privileged subgroup within this demographic column
    draw <class 'bool'>: boolean of whether to draw the plot or not
    metrics <class 'list'>: list of strings of shorthand names of metrics to make graphs of
    graphpath <class 'str'>: string of folder path to where the plot should be saved as a png. If None then the plot
    will not be saved
    Returns
    aplot <class 'plotnine.ggplot.ggplot'>: plotnine plot of the metrics and demographics chosen

Function to draw two graphs of the Receiver Operating Characteristic curves for a demographic column
measured.RocPlots(colname, 
        draw=True, 
        graphpath=None)
    colname <class 'str'>: string of demographic column name
    draw <class 'bool'>: boolean of whether to draw the graphs or not
    graphpath <class 'str'>: string of folder path to where the plot should be saved as a png. If None then the plot
    will not be saved
    Returns
    aplot <class 'plotnine.ggplot.ggplot'>: plotnine graph of the ROC curve
    rocgraph2 <class 'plotnine.ggplot.ggplot'>: plotnine graph of the ROC curve zoomed in the upper left hand quadrant

Function to print out all chosen metrics for all chosen demographic columns in a table
measured.PrintMetrics(columnlist=[], 
            metrics=['STP', 'TPR', 'PPV', 'FPR', 'ACC'])
    columnlist <class 'list'>: list of strings of the names of the demographic columns to print metrics for
    metrics <class 'list'>: list of shorthand names of the metrics to print out

Function to calculate the ratio of metric values for one demographic column
measured.PrintRatios(colname, 
            privileged, 
            metrics=['STP', 'TPR', 'PPV', 'FPR', 'ACC'], 
            printout=True)
    colname <class 'str'>: string of demographic column name
    privileged <class 'str'>: string of name for the privileged subgroup within this demographic column
    metrics <class 'list'>: list of strings of shorthand names of metrics to calculate ratios for
    printout <class 'bool'>: boolean of whether or not to print out the table of ratios calculated
    Returns
    metricsdf <class 'pandas.core.frame.DataFrame'>: table of the ratios calculated

measure_disparity_methods.measured also has a class variable fullnames which is a dictionary wherein the shorthand names are the
keys and the full length names of the metrics are the values
fullnames = {
        'STP': 'Statistical Parity',
        'TPR': 'Equal Opportunity',
        'PPV': 'Predictive Parity',
        'FPR': 'Predictive Equality',
        'ACC': 'Accuracy',
        'TNR': 'True Negative Rate',
        'NPV': 'Negative Predictive Value',
        'FNR': 'False Negative Rate',
        'FDR': 'False Discovery Rate',
        'FOR': 'False Omission Rate',
        'TS': 'Threat Score',
        'FS': 'F1 Score'
    }
    Dictionary of metric 

######################
##Bias Mitigation Tool
######################
CLI tool usage: Sample input for mitigate_disparity.py: 

usage: mitigate_disparity.py [-h] [-a] [-f] [-p] settings.ini 

Settings	:	settings.ini file 
-h, --help	:	show this help message and exit 
-a, --adjust	:	Model Agnostic Adjustment (default: False) 
-f, --fit	:	Fit Debiased Model Thresholds (default: False) 
-p, -predict	:	Fit Debiased Model and Predict (default: False) 

The settings ini file for -a option must include [dataset information], [paths], and [other]. The settings ini file for
the options -f and -p must include [model settings] and [model arguments] in addition to [dataset information], 
[paths], and [other].

Sample CLI usages:
python mitigate_disparity.py -a settings-agnostic.ini
python mitigate_disparity.py -f settings-model.ini
python mitigate_disparity.py -p settings-model.ini
python mitigate_disparity.py -h

Sample settings ini file:
[model settings]
model_module: xgboost
model_class: XGBClassifier
fit_method: fit
predict_method: predict
predict_proba_method: predict_proba

[model arguments]
eta: 0.1
subsample: 0.7
max_depth: 8
scale_pos_weight: 5 

[dataset information]
target_variable: esrd_flag
input_variables: age,body_mass_index,body_weight,calcium,chloride,creatinine,urea_nitrogen,diabetes_flag
demo_variables: RACE,GENDER,INCOME
advantaged_groups: white,M,High
probability_column: proba
predicted_column: predicted
weight_column: n/a

[paths]
input_path: ~/data/modeling_dataset.csv
predict_input_path: ~/data/modeling_dataset.csv
output_path: debiased_predicted_dataset.csv

[other]
optimization_repetitions: 2
display_metrics: STP,TPR,PPV,FPR,ACC

Besides the CLI the tool can also be more directly used and customized by importing mitigate_disparity_methods. The 
main functions of mitigate_disparity_methods are described below.

model_agnostic_adjustment(modeling_dataset, 
                            y, 
                            proba, 
                            D, 
                            demo_dict, 
                            reps = 30)
    modeling_dataset <class 'pandas.core.frame.DataFrame'>: data from another model to be debiased
    y <class 'pandas.core.series.Series'>: series of actual values, either 0 or 1, from the input data
    proba <class 'pandas.core.series.Series'>: series of probabilities 0 to 1, from the input data
    D <class 'pandas.core.frame.DataFrame'>: dataframe of demographic values with one row for each observation
    demo_dict <class 'dict'>: dictionary with the names of the demographic columns as keys and the names of the 
    privileged subgroups as values
    reps <class 'int'>: number of repetitions that this function will run through to create debiased predictions
    Returns
    output <class 'pandas.core.frame.DataFrame'>: data from another model but with new debiased predictions added under
    the column "debiased_prediction"

mitigate_disparity_methods.debiased_model(config)
    config <class 'configparser.ConfigParser'>: object with all the configuration settings to initialize an object of
    the debiased_model class

debiased_model.fit(X, 
                    y, 
                    D, 
                    demo_dict, 
                    reps = 30, 
                    *args, 
                    **kwargs)
    X <class 'pandas.core.frame.DataFrame'>: data of the various input variables used to train the model
    y  <class 'pandas.core.series.Series'>: series of actual values from the modelling input data
    D <class 'pandas.core.frame.DataFrame'>: dataframe of demographic values with one row for each observation
    demo_dict <class 'dict'>: dictionary with the names of the demographic columns as keys and the names of the 
    privileged subgroups as values
    reps <class 'int'>: number of repetitions the program will go through to create the new debiased cutoff thresholds
    args, kwargs: any number of arguments to be passed through from [model settings] and [model arguments] to whatever
    chosen module the users wishes to use for their machine learning model

debiased_model.predict(X, 
                        D)
    X <class 'pandas.core.frame.DataFrame'>: data of the various input variables used to train the model
    D <class 'pandas.core.frame.DataFrame'>: dataframe of demographic values with one row for each observation
    Returns
    prediction_df <class 'pandas.core.frame.DataFrame'>: dataframe of just one column containing the new predicted 
    values for each observation

debiased_model.get_debiased_thresholds()
    Returns
    debiased_thresholds <class 'pandas.core.frame.DataFrame'>: new cutoff thresholds calculated to be debiased across
    all the demographic subgroups and interactions between subgroups
