
# or you can directly load the cleaned & imputed dataset : 
# 	cell_samples_clean.csv
df = pd.read_csv(r'F:\OneDrive\Internet\computer\Datasets\cell_samples.csv')

############
###########

# pre-processing

# checking for any missing values throught the dataframe.
# if numeric data is converted to string, missing values would not show here. see below.
df.isnull().sum().sum()

# to render the results for each individual column separately :
df.isnull().sum()


x = df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
y = df['Class']

y.unique()
# Out[48]: array([2, 4], dtype=int64)

# unlike above, here the missing values would show up.
x.isnull().sum()

#######

# how do you kow that '?' is the source of the problem ?
# 	 error report when trying to run standard-scaler on the raw data.
x.eq('?').any()
x['BareNuc'].eq('?')  # it's a boolean_selector (mask).
x['BareNuc'][ x['BareNuc'].eq('?') ]

##########

# this will have 2 effects : 
# changing the string nnumbers ( '1' ) to real digits.
# changing the unidentified data ( '?' ) to nan.
x['BareNuc']  = pd.to_numeric( x['BareNuc'] , errors='coerce')

###########

# imputation of missing values.

# it : imputation strategy
it = si( missing_values=pd.NA , strategy='mean') 

# the output can not be performed in-place  => define a new variable.
# the output is a nunmpy array [ converts a dataframe to an np.array ]
# input : should be 2-dimensional ! : a 2D array or a dataframe { not a pd.series}.
out = it.fit_transform(x['BareNuc'].to_frame())

# this throughs a SettingWithCopyWarning.
# this warning is invalied in my case : the reason is the 1st phrase : x['BareNuc'] :
# this is a copy of df[['Clump', 'UnifSize', ... ]
# the original df will not be affected : but we don't care : we don't need it !
x['BareNuc'] = out

#####

# to save te clean data
df_clean = pd.concat( [x,y] , axis=1)

df_clean.to_csv(r'F:\OneDrive\Internet\computer\Datasets\cell_samples_clean.csv')

#########
#########

x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.5)

pipe = mpi( stsc() , SVC() )

# type it to get the automatic names assigned by scikit, : (for the next step).
pipe
# Out[38]: Pipeline(steps=[('standardscaler', StandardScaler()), ('svc', SVC())])


pipe.set_params(svc__gamma='auto')


##########
##########

pipe.fit(x_train, y_train)

y_test_pred = pipe.predict(x_test)

# since your array is not binary (0,1), you should set the 'average' to None.
# otherwise it throes the error mentipoend in the svm.doc.s
js(y_test , y_test_pred , average=None)
cf_m(y_test , y_test_pred )  #  confusion matrix.
cf_m_d.from_predictions(y_test , y_test_pred )  #  display _ confusion matrix.
