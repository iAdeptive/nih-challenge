## nih-challenge
bias detection challenge

####################### <br>
##Bias Measurement Tool <br>
####################### <br>
CLI tool usage: measure_disparity.py [-h] settings.ini <br>

Settings	:	settings.ini file <br>
-h, --help	:	show this help message and exit <br>

The settings ini file for must include [dataset information], [paths], and [other]. <br>

Sample CLI usages: <br>
python measure_disparity.py settings-measure.ini <br>
python measure_disparity.py -h <br>

Sample settings ini file: <br>
[dataset information] <br>
target_variable: actual <br>
demo_variables: RACE,GENDER,INCOME <br>
advantaged_groups: white,M,High <br>
probability_column: proba <br>
predicted_column: predicted <br>
weight_column: n/a <br>

[paths] <br>
input_path: ~/data/model_results.csv <br>

[other] <br>
display_metrics: STP,TPR,PPV,FPR,ACC <br>

Besides the CLI the tool can also be more directly used and customized by importing measure_disparity_methods. The <br> 
main functions of measure_disparity_methods are described below. <br>

Function to intialize a measured object from measure_disparity_methods.py <br>
measure_disparity_methods.measured(dataset, <br>
        democols, <br>
        intercols, <br>
        actualcol='actual', <br>
        predictedcol='predicted', <br>
        probabilitycol='probability', <br>
        weightscol='weights') <br> 
    dataset <class 'pandas.core.frame.DataFrame'>: data generated from a model you which to analyze <br>
    democols <class 'list'>: list of strings of demographic column names <br>
    intercols <class 'list'>: list of lists with pairs of strings of demo column names you wish to see <br> 
    interactions between <br>
    actualcol <class 'str'>: string of column name with actual values formatted as 0 or 1 <br>
    predictedcol <class 'str'>: string of column name with predicted values formatted as 0 or 1 <br>
    probabilitycol <class 'str'>: string of column name with probability values formatted as floats 0 to 1 <br>
    weightscol <class 'str'>: string of column name with weights formatted as ints or floats <br>
    Returns <br>
    measured <class 'measure_disparity.measured'>: object of the class from measure_disparity.py <br>

Function to draw a plot of any number of given metrics for a given demographic column <br>
measured.MetricPlots(colname,  <br>
            privileged, <br>
            draw=True, <br>
            metrics=['STP', 'TPR', 'PPV', 'FPR', 'ACC'], <br>
            graphpath=None) <br>
    colname <class 'str'>: string of demographic column name <br>
    privileged <class 'str'>: string of name for the privileged subgroup within this demographic column <br>
    draw <class 'bool'>: boolean of whether to draw the plot or not <br>
    metrics <class 'list'>: list of strings of shorthand names of metrics to make graphs of <br>
    graphpath <class 'str'>: string of folder path to where the plot should be saved as a png. If None then the <br>
    plot will not be saved
    Returns <br>
    aplot <class 'plotnine.ggplot.ggplot'>: plotnine plot of the metrics and demographics chosen <br>

Function to draw two graphs of the Receiver Operating Characteristic curves for a demographic column <br>
measured.RocPlots(colname, <br>
        draw=True, <br>
        graphpath=None) <br>
    colname <class 'str'>: string of demographic column name <br>
    draw <class 'bool'>: boolean of whether to draw the graphs or not <br>
    graphpath <class 'str'>: string of folder path to where the plot should be saved as a png. If None then the 
    plot will not be saved <br>
    Returns <br>
    aplot <class 'plotnine.ggplot.ggplot'>: plotnine graph of the ROC curve <br>
    rocgraph2 <class 'plotnine.ggplot.ggplot'>: plotnine graph of the ROC curve zoomed in on the upper left hand <br>
    quadrant <br>

Function to print out all chosen metrics for all chosen demographic columns in a table <br>
measured.PrintMetrics(columnlist=[], <br>
            metrics=['STP', 'TPR', 'PPV', 'FPR', 'ACC']) <br>
    columnlist <class 'list'>: list of strings of the names of the demographic columns to print metrics for <br>
    metrics <class 'list'>: list of shorthand names of the metrics to print out <br>

Function to calculate the ratio of metric values for one demographic column <br>
measured.PrintRatios(colname,  <br>
            privileged, <br>
            metrics=['STP', 'TPR', 'PPV', 'FPR', 'ACC'], <br>
            printout=True) <br>
    colname <class 'str'>: string of demographic column name <br>
    privileged <class 'str'>: string of name for the privileged subgroup within this demographic column <br>
    metrics <class 'list'>: list of strings of shorthand names of metrics to calculate ratios for <br>
    printout <class 'bool'>: boolean of whether or not to print out the table of ratios calculated <br>
    Returns <br>
    metricsdf <class 'pandas.core.frame.DataFrame'>: table of the ratios calculated <br>

measure_disparity_methods.measured also has a class variable fullnames which is a dictionary wherein the <br>
shorthand names are the keys and the full length names of the metrics are the values <br>
fullnames = { <br>
        'STP': 'Statistical Parity', <br>
        'TPR': 'Equal Opportunity', <br>
        'PPV': 'Predictive Parity', <br>
        'FPR': 'Predictive Equality', <br>
        'ACC': 'Accuracy', <br>
        'TNR': 'True Negative Rate', <br>
        'NPV': 'Negative Predictive Value', <br>
        'FNR': 'False Negative Rate', <br>
        'FDR': 'False Discovery Rate', <br>
        'FOR': 'False Omission Rate', <br>
        'TS': 'Threat Score', <br>
        'FS': 'F1 Score' <br>
    } <br>

###################### <br>
##Bias Mitigation Tool <br>
###################### <br>
CLI tool usage: Sample input for mitigate_disparity.py:  <br>

usage: mitigate_disparity.py [-h] [-a] [-f] [-p] settings.ini  <br>

Settings	:	settings.ini file  <br>
-h, --help	:	show this help message and exit  <br>
-a, --adjust	:	Model Agnostic Adjustment (default: False)  <br>
-f, --fit	:	Fit Debiased Model Thresholds (default: False)  <br>
-p, -predict	:	Fit Debiased Model and Predict (default: False)  <br>

The settings ini file for -a option must include [dataset information], [paths], and [other]. The settings ini <br>
file for the options -f and -p must include [model settings] and [model arguments] in addition to [dataset  <br>
information], [paths], and [other]. <br>

Sample CLI usages: <br>
python mitigate_disparity.py -a settings-agnostic.ini <br>
python mitigate_disparity.py -f settings-model.ini <br>
python mitigate_disparity.py -p settings-model.ini <br>
python mitigate_disparity.py -h <br>

Sample settings ini file: <br>
[model settings] <br>
model_module: xgboost <br>
model_class: XGBClassifier <br>
fit_method: fit <br>
predict_method: predict <br>
predict_proba_method: predict_proba <br>

[model arguments] <br>
eta: 0.1 <br>
subsample: 0.7 <br>
max_depth: 8 <br>
scale_pos_weight: 5  <br>

[dataset information] <br>
target_variable: esrd_flag <br>
input_variables: age,body_mass_index,body_weight,calcium,chloride,creatinine,urea_nitrogen,diabetes_flag <br>
demo_variables: RACE,GENDER,INCOME <br>
advantaged_groups: white,M,High <br>
probability_column: proba <br>
predicted_column: predicted <br>
weight_column: n/a <br>

[paths] <br>
input_path: ~/data/modeling_dataset.csv <br>
predict_input_path: ~/data/modeling_dataset.csv <br>
output_path: debiased_predicted_dataset.csv <br>

[other] <br>
optimization_repetitions: 2 <br>
display_metrics: STP,TPR,PPV,FPR,ACC <br>

Besides the CLI the tool can also be more directly used and customized by importing mitigate_disparity_methods.  <br>
The main functions of mitigate_disparity_methods are described below. <br>

model_agnostic_adjustment(modeling_dataset,  <br>
                            y,  <br>
                            proba,  <br>
                            D,  <br>
                            demo_dict,  <br>
                            reps = 30) <br>
    modeling_dataset <class 'pandas.core.frame.DataFrame'>: data from another model to be debiased <br>
    y <class 'pandas.core.series.Series'>: series of actual values, either 0 or 1, from the input data <br>
    proba <class 'pandas.core.series.Series'>: series of probabilities 0 to 1, from the input data <br>
    D <class 'pandas.core.frame.DataFrame'>: dataframe of demographic values with one row for each observation <br>
    demo_dict <class 'dict'>: dictionary with the names of the demographic columns as keys and the names of the  <br>
    privileged subgroups as values <br>
    reps <class 'int'>: number of repetitions that this function will run through to create debiased predictions <br>
    Returns
    output <class 'pandas.core.frame.DataFrame'>: data from another model but with new debiased predictions added <br>
    under the column "debiased_prediction" <br>

mitigate_disparity_methods.debiased_model(config) <br>
    config <class 'configparser.ConfigParser'>: object with all the configuration settings to initialize an <br>
    object of the debiased_model class <br>

debiased_model.fit(X,  <br>
                    y,  <br>
                    D,  <br>
                    demo_dict,  <br>
                    reps = 30,  <br>
                    *args,  <br>
                    **kwargs) <br>
    X <class 'pandas.core.frame.DataFrame'>: data of the various input variables used to train the model <br>
    y  <class 'pandas.core.series.Series'>: series of actual values from the modelling input data <br>
    D <class 'pandas.core.frame.DataFrame'>: dataframe of demographic values with one row for each observation <br>
    demo_dict <class 'dict'>: dictionary with the names of the demographic columns as keys and the names of the  <br>
    privileged subgroups as values
    reps <class 'int'>: number of repetitions the program will go through to create the new debiased cutoff <br>
    thresholds args, kwargs: any number of arguments to be passed through from [model settings] and [model <br>
    arguments] to whatever chosen module the users wishes to use for their machine learning model <br>

debiased_model.predict(X,  <br>
                        D) <br>
    X <class 'pandas.core.frame.DataFrame'>: data of the various input variables used to train the model <br>
    D <class 'pandas.core.frame.DataFrame'>: dataframe of demographic values with one row for each observation <br>
    Returns <br>
    prediction_df <class 'pandas.core.frame.DataFrame'>: dataframe of just one column containing the new <br>
    predicted values for each observation <br>

debiased_model.get_debiased_thresholds() <br>
    Returns <br>
    debiased_thresholds <class 'pandas.core.frame.DataFrame'>: new cutoff thresholds calculated to be debiased <br>
    across all the demographic subgroups and interactions between subgroups <br>
