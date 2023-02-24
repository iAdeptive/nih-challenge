##measure_disparity_methods.py for iAdeptive

import pandas as pd
from plotnine import *
from sklearn.metrics import roc_curve
from os import makedirs

class measured:
    fullnames = { #dictionary of full length names for each of the 12 metric functions
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

    def __init__(self, 
                 dataset,
                 democols,
                 intercols,
                 actualcol='actual',
                 predictedcol='predicted',
                 probabilitycol='probability',
                 weightscol='weights'):
        #read in inputs
        self.df = dataset.copy()
        self.democols = democols
        self.actcol = actualcol
        self.predcol = predictedcol
        self.probcol = probabilitycol
        self.wcol = weightscol

        self.tpcount = {} #dictionary for true positive count
        self.fpcount = {}
        self.fncount = {}
        self.tncount = {}

        #create column for subgroup interactions if intercols is not an empty list
        if intercols:
            self._intergroups(intercols)

        #create column with truth values
        self.df['truths'] = self.df[self.actcol] + (self.df[self.predcol] * 2)

        #calculating truth counts per demographic
        self._calc()

        #dictionary of shorthand names for each of the 12 metric functions
        self.metricnames = {
            'STP': self.StatParity,
            'TPR': self.EqualOpp,
            'PPV': self.PredParity,
            'FPR': self.PredEqual,
            'ACC': self.Accuracy,
            'TNR': self.TrueNeg,
            'NPV': self.NegPV,
            'FNR': self.FalseNeg,
            'FDR': self.FalseDis,
            'FOR': self.FalseOm,
            'TS': self.ThreatScore,
            'FS': self.FScore
        }
    
    #calculating truth counts for each demographic
    def _calc(self):
        for col in self.df:
            if col in self.democols:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    adf = self.df[self.df[col] == key]
                    if self.wcol in adf.columns:
                        tpdf = adf[adf.truths == 3]
                        fpdf = adf[adf.truths == 2]
                        fndf = adf[adf.truths == 1]
                        tndf = adf[adf.truths == 0]
                        self.tpcount[key] = sum(tpdf.weights)
                        self.fpcount[key] = sum(fpdf.weights)
                        self.fncount[key] = sum(fndf.weights)
                        self.tncount[key] = sum(tndf.weights)
                    else:
                        self.tpcount[key] = sum(adf.truths == 3)
                        self.fpcount[key] = sum(adf.truths == 2)
                        self.fncount[key] = sum(adf.truths == 1)
                        self.tncount[key] = sum(adf.truths == 0)
    
    #create new intersectional columns with pairs of subgroups
    def _intergroups(self, intercols):
        for pair in intercols:
            pairname = pair[0] + '-' + pair[1]
            self.df[pairname] = self.df[pair[0]] + '-' + self.df[pair[1]]
            self.democols.append(pairname) #adding in intersectional columns
    
    #converting the output to a dataframe format
    def _todf(self, adict, shortname):
        adf = pd.DataFrame.from_dict(adict, orient='index', columns=[self.fullnames[shortname]])
        adf.reset_index(inplace=True)
        adf = adf.rename(columns={'index': 'Subgroup'})
        return adf
    
    #printing out a table with the requested metrics
    @staticmethod
    def _printtable(mname, metricdict):
        dlen = len('Subgroup') + 1
        for key in metricdict:
            if len(key) > dlen:
                dlen = len(key) + 1 #setting length of first column for the table
        print('{0:<{1}}|{2:>{3}}'.format('Subgroup',dlen,mname, len(mname)))
        for item in metricdict:
            print('{0:<{1}}|{2:>{3}}'.format(item,dlen,metricdict[item], len(mname)))
    
    #outputting a figure with graphs of all the metrics for a subgroup category
    def MetricPlots(self, 
                    colname, 
                    privileged, 
                    draw=True, 
                    metrics=['STP', 'TPR', 'PPV', 'FPR', 'ACC'], 
                    graphpath=None):
        metricsdf = pd.DataFrame()
        for name in self.metricnames:
            if name in metrics:
                if metricsdf.empty:
                    metricsdf = self.metricnames[name](colname)
                else:
                    metricsdf[self.fullnames[name]] = self.metricnames[name](colname).iloc[:,1]
        demokeys = self.df[colname].unique().tolist()
        metricsdf = metricsdf.set_index('Subgroup')
        metricsdf.index.name = None
        metricsdf = metricsdf.loc[demokeys, :]
        metricsdf = metricsdf.transpose()
        demokeys.remove(privileged)
        for demo in demokeys:
            metricsdf[demo] = metricsdf[demo] / metricsdf[privileged] - 1
        metricsdf.reset_index(inplace=True)
        gheight = len(demokeys) * len(metrics) * 0.5
        if len(demokeys) > 1:
            metricsdf = pd.melt(metricsdf, id_vars='index', value_vars=demokeys)
            atitle = 'Fairness Metrics by ' + colname
            aplot = (ggplot(metricsdf)
            + aes(x='index', 
                  y='value', 
                  fill='index')
            + geom_bar(stat='identity', 
                       position='stack', 
                       show_legend=False)
            + scale_y_continuous(limits= (-1, 1), 
                                 breaks= [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1], 
                                 labels= (0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2))
            + theme(figure_size= (12, gheight))
            + coord_flip()
            + facet_wrap('variable', 
                         nrow=len(demokeys))
            + annotate('polygon',
                       x=[0,0,13,13],
                       y=[-0.2,-1,-1,-0.2], 
                       fill='#D9544D', 
                       alpha=0.2)
            + annotate('polygon', 
                       x=[0,0,13,13], 
                       y=[0.2,-0.2,-0.2,0.2], 
                       fill='#98FB98', 
                       alpha=0.2)
            + annotate('polygon', 
                       x=[0,0,13,13], 
                       y=[0.2,1,1,0.2], 
                       fill='#D9544D', 
                       alpha=0.2)
            + labs(y='Score Ratio', 
                   x='Metric', 
                   title=atitle)
            )
        else: #when only two demographics to compare
            atitle = 'Fairness Metrics ' + demokeys[0] + ' vs ' + privileged
            aplot = (ggplot(metricsdf)
            + aes(x='index', 
                  y=demokeys[0])
            + geom_bar(stat='identity', 
                       position='stack')
            + scale_y_continuous(limits= (-1, 1), 
                                 breaks= [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1], 
                                 labels = (0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2))
            + theme(figure_size = (12, gheight))
            + coord_flip()
            + annotate('polygon', 
                       x=[0,0,13,13], 
                       y=[-0.2,-1,-1,-0.2], 
                       fill='#D9544D', 
                       alpha=0.2)
            + annotate('polygon', 
                       x=[0,0,13,13], 
                       y=[0.2,-0.2,-0.2,0.2], 
                       fill='#98FB98', 
                       alpha=0.2)
            + annotate('polygon', 
                       x=[0,0,13,13], 
                       y=[0.2,1,1,0.2], 
                       fill='#D9544D', 
                       alpha=0.2)
            + labs(y='Score Ratio', 
                   x='Metric', 
                   title=atitle)
            )
        if draw:
            print(aplot)
        if graphpath != None:
            makedirs(graphpath, exist_ok=True)
            fname = 'FairnessMetrics'
            for item in metrics:
                fname += item
            fname += 'by' + colname + '.png'
            aplot.save(filename=fname, path=graphpath)
        return aplot
    
    #outputting the receiver operating characteristic curves for one category of subgroups
    def RocPlots(self, 
                 colname, 
                 draw=True, 
                 graphpath=None):
        rocdf = pd.DataFrame(columns=['subgroup', 'fpr', 'tpr'])
        for subgroup in self.df[colname].unique():
            adf = self.df[self.df[colname] == subgroup]
            fpr, tpr, thresholds = roc_curve(adf[self.actcol], adf[self.probcol])
            rdf = pd.DataFrame()
            rdf['fpr'] = fpr
            rdf['tpr'] = tpr
            rdf['subgroup'] = subgroup
            rocdf = pd.concat([rocdf, rdf])
        atitle = 'Receiver Operating Characteristic Curves by ' + colname
        aplot = (ggplot(rocdf)
        + aes(x='fpr', 
              y='tpr', 
              color='subgroup')
        + geom_line()
        + xlim(0, 1)
        + ylim(0, 1)
        + labs(y='True Positive Rate', 
               x='False Positive Rate', 
               title=atitle)
        )
        atitle = 'Zoomed In ROC Curves by ' + colname
        rocgraph2 = (ggplot(rocdf)
        + aes(x='fpr', 
              y='tpr', 
              color='subgroup')
        + geom_line()
        + xlim(0, 0.5)
        + ylim(0.5, 1)
        + labs(y='True Positive Rate', 
               x='False Positive Rate', 
               title=atitle)
        )
        if draw:
            print(aplot)
            print(rocgraph2)
        if graphpath != None:
            makedirs(graphpath, exist_ok=True)
            fname = 'ROCcurvesby' + colname + '.png'
            aplot.save(filename=fname, path=graphpath)
            fname = 'ZoomedInROCcurvesby' + colname + '.png'
            rocgraph2.save(filename=fname, path=graphpath)
        return [aplot, rocgraph2]
    
    #printing out all the chosen metrics in a table
    def PrintMetrics(self, 
                     columnlist=[], 
                     metrics=['STP', 'TPR', 'PPV', 'FPR', 'ACC']):
        metricsdf = pd.DataFrame()
        if not columnlist:
            columnlist = self.democols
        for name in self.metricnames:
            if name in metrics:
                if metricsdf.empty:
                    metricsdf = self.metricnames[name](columnlist)
                else:
                    adf = self.metricnames[name](columnlist)
                    metricsdf[self.fullnames[name]] = adf.iloc[:,1]
        megalist = []
        for col in metricsdf.columns:
            megalist.append(metricsdf[col].array.tolist())
        megalist = list(map(list, zip(*megalist)))
        demolen = 0
        for item in metricsdf['Subgroup']: #setting length of 1st column to longest subgroup name
            if len(item) > demolen:
                demolen = len(item)
        col_widths = [demolen]
        metricsdf = metricsdf.set_index('Subgroup') #moving Subgroups column back to the index
        for col in metricsdf.columns:
            col_widths.append(len(col))
        formatted = ' '.join(['%%%ds |' % (width + 1) for width in col_widths ])[:-1]
        metricsdf.reset_index(inplace=True) #moving Subgroups column from index into its own column
        colnames = metricsdf.columns

        #printing out the table
        print(formatted % tuple(colnames))
        for row in megalist:
            print(formatted % tuple(row))
    
    #printing out all the metrics as ratios for one demographic column
    def PrintRatios(self, 
                    colname, 
                    privileged, 
                    metrics=['STP', 'TPR', 'PPV', 'FPR', 'ACC'], 
                    printout=True):
        metricsdf = pd.DataFrame()
        for name in self.metricnames:
            if name in metrics:
                if metricsdf.empty:
                    metricsdf = self.metricnames[name](colname)
                else:
                    adf = self.metricnames[name](colname)
                    metricsdf[self.fullnames[name]] = adf.iloc[:,1]
        metricsdf = metricsdf.set_index('Subgroup')
        for colname in metricsdf.columns:
            metricsdf = metricsdf.rename(columns={colname: (colname + ' ratio')})
        for rowname in metricsdf.index:
            if rowname != privileged:
                metricsdf.loc[rowname,:] = round(metricsdf.loc[rowname,:] / metricsdf.loc[privileged,:], 4)
        metricsdf = metricsdf.drop(privileged, axis=0) #dropping out the privileged group from this df of ratios
        metricsdf.reset_index(inplace=True) #moving Subgroups column from index into its own column
        if printout:
            megalist = []
            for col in metricsdf.columns:
                megalist.append(metricsdf[col].array.tolist())
            megalist = list(map(list, zip(*megalist)))
            demolen = len('Subgroup')
            for item in metricsdf['Subgroup']: #setting length of 1st column to longest subgroup name
                if len(item) > demolen:
                    demolen = len(item)
            if len(metrics) < 4:
                col_widths = [demolen]
                metricsdf = metricsdf.set_index('Subgroup') #moving Subgroups column back to the index
                for col in metricsdf.columns:
                    col_widths.append(len(col))
                metricsdf.reset_index(inplace=True) #moving Subgroups column from index into its own column
                formatted = ' '.join(['%%%ds |' % (width + 1) for width in col_widths ])[:-1]
                colnames = metricsdf.columns
                #printing out table of ratios
                print(formatted % tuple(colnames))
                for row in megalist:
                    print(formatted % tuple(row))
            else: #when there are alot of metrics in the outputted table
                colwidth = 0
                for col in metricsdf.columns:
                    if len(col) > colwidth:
                        colwidth = len(col) + 1
                formatted = ' %%%ds | %%%ds' % (demolen, colwidth)   
                for i in range(len(metrics)): #cycling through all the metrics
                    print(formatted % ('Subgroup', metricsdf.columns[i+1]))
                    for row in megalist:
                        arow = (row[0], row[i+1])
                        print(formatted % arow)
                    print('\n')
        return metricsdf
        

    #calculate Equal Opportunity metric aka True Positive Rate
    def EqualOpp(self, columnlist=[], printout=False):
        tprs = {}
        #if columns are not specified then show metric for all columns
        if not columnlist: 
            columnlist = self.democols
        #if there was only one input and the columnlist isn't a list, then make it a list
        elif not isinstance(columnlist, list): 
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.tpcount[key] + self.fncount[key]) == 0:
                        tprs[key] = 0
                    else:
                        tprs[key] = round((self.tpcount[key] / (self.tpcount[key] + self.fncount[key])), 4)
        if printout: #print out a table if the user specifies to do so
            self._printtable(self.fullnames['TPR'], tprs)
        return self._todf(tprs, 'TPR') #export output as a dataframe
    
    #calculate Predictive Equality metric aka False Positive Rate
    def PredEqual(self, columnlist=[], printout=False):
        fprs = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.fpcount[key] + self.tncount[key]) == 0:
                        fprs[key] = 0
                    else:
                        fprs[key] = round(self.fpcount[key] / (self.fpcount[key] + self.tncount[key]), 4)
        if printout:
            self._printtable(self.fullnames['FPR'], fprs)
        return self._todf(fprs, 'FPR')
    
    #calculate Statistical Parity metric aka predicted value per subgroup
    def StatParity(self, columnlist=[], printout=False):
        sparity = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    sparity[key] = round((self.tpcount[key] + self.fpcount[key]) / 
                                         (self.tpcount[key] + self.fpcount[key] + 
                                          self.fncount[key] + self.tncount[key]), 4)
        if printout:
            self._printtable(self.fullnames['STP'], sparity)
        return self._todf(sparity, 'STP')
    
    #calculate Predictive Parity metric aka Positive Predictive Value
    def PredParity(self, columnlist=[], printout=False):
        pparity = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.tpcount[key] + self.fpcount[key]) == 0:
                        pparity[key] = 0
                    else:
                        pparity[key] = round(self.tpcount[key] / (self.tpcount[key] + self.fpcount[key]), 4)
        if printout:
            self._printtable(self.fullnames['PPV'], pparity)
        return self._todf(pparity, 'PPV')
    
    #calculate Accuracy metric
    def Accuracy(self, columnlist=[], printout=False):
        acc = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    acc[key] = round((self.tpcount[key] + self.tncount[key]) / 
                                     (self.tpcount[key] + self.fpcount[key] + 
                                      self.fncount[key] + self.tncount[key]), 4)
        if printout:
            self._printtable(self.fullnames['ACC'], acc)
        return self._todf(acc, 'ACC')
    
    #calculate True Negative Rate
    def TrueNeg(self, columnlist=[], printout=False):
        tnrs = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.tncount[key] + self.fpcount[key]) == 0:
                        tnrs[key] = 0
                    else:
                        tnrs[key] = round(self.tncount[key] / (self.tncount[key] + self.fpcount[key]), 4)
        if printout:
            self._printtable(self.fullnames['TNR'], tnrs)
        return self._todf(tnrs, 'TNR')
    
    #calculate Negative Predictive Value
    def NegPV(self, columnlist=[], printout=False):
        npv = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.tncount[key] + self.fncount[key]) == 0:
                        npv[key] = 0
                    else:
                        npv[key] = round(self.tncount[key] / (self.tncount[key] + self.fncount[key]), 4)
        if printout:
            self._printtable(self.fullnames['NPV'], npv)
        return self._todf(npv, 'NPV')
    
    #calculate False Negative Rate
    def FalseNeg(self, columnlist=[], printout=False):
        fnrs = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.fncount[key] + self.tpcount[key]) == 0:
                        fnrs[key] = 0
                    else:
                        fnrs[key] = round(self.fncount[key] / (self.fncount[key] + self.tpcount[key]), 4)
        if printout:
            self._printtable(self.fullnames['FNR'], fnrs)
        return self._todf(fnrs, 'FNR')
    
    #calculate False Discovery Rate
    def FalseDis(self, columnlist=[], printout=False):
        fdrs = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.fpcount[key] + self.tpcount[key]) == 0:
                        fdrs[key] = 0
                    else:
                        fdrs[key] = round(self.fpcount[key] / (self.fpcount[key] + self.tpcount[key]), 4)
        if printout:
            self._printtable(self.fullnames['FDR'], fdrs)
        return self._todf(fdrs, 'FDR')
    
    #calculate False Omission Rate
    def FalseOm(self, columnlist=[], printout=False):
        fors = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.fncount[key] + self.tncount[key]) == 0:
                        fors[key] = 0
                    else:
                        fors[key] = round(self.fncount[key] / (self.fncount[key] + self.tncount[key]), 4)
        if printout:
            self._printtable(self.fullnames['FOR'], fors)
        return self._todf(fors, 'FOR')
    
    #calculate Threat Score
    def ThreatScore(self, columnlist=[], printout=False):
        threats = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.tpcount[key] + self.fncount[key] + self.fpcount[key]) == 0:
                        threats[key] = 0
                    else:
                        threats[key] = round(self.tpcount[key] /
                                             (self.tpcount[key] + self.fncount[key] + self.fpcount[key]), 4)
        if printout:
            self._printtable(self.fullnames['TS'], threats)
        return self._todf(threats, 'TS')
    
    #calculate F1 Score
    def FScore(self, columnlist=[], printout=False):
        fscores = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    ppv = self.PredParity()
                    pvalue = ppv.loc[ppv['Subgroup'] == key, self.fullnames['PPV']].iloc[0]
                    tpr = self.EqualOpp()
                    tvalue = tpr.loc[tpr['Subgroup'] == key, self.fullnames['TPR']].iloc[0]
                    if (pvalue + tvalue) == 0:
                        fscores[key] = 0
                    else:
                        fscores[key] = round(2 * pvalue * tvalue / (pvalue + tvalue), 4)
        if printout:
            self._printtable(self.fullnames['FS'], fscores)
        return self._todf(fscores, 'FS')

    #output dataframe table with more human readable truth column
    def ReadTruths(self):
        df2 = self.df.copy()
        df2.loc[df2['truths'] == 3, 'truths'] = 'True Positive'
        df2.loc[df2['truths'] == 2, 'truths'] = 'False Positive'
        df2.loc[df2['truths'] == 1, 'truths'] = 'False Negative'
        df2.loc[df2['truths'] == 0, 'truths'] = 'True Negative'
        return df2

    
