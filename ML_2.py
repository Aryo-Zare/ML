

# 1st explore.
# at least 6 significant sois.
# here, mistakenly 2p data were selected on 3p error correected data. 
    # not a big deal for ML.

plt.scatter( df_6sig_8soi_3p['Tau_all'] , df_6sig_8soi_3p['A_all'] )
plt.xscale('log')
plt.yscale('log')

plt.savefig( r'/home/azare/groups/PrimNeu/Aryo/analysis/test/2.pdf')

######

df_6sig_8soi_3p[[ 'Tau_all' , 'A_all' ]][ :4]

df_6sig_8soi_3p_log10 = np.log10( df_6sig_8soi_3p[[ 'Tau_all' , 'A_all' ]] )
        
        # df_6sig_8soi_3p_log10.shape
        # Out[814]: (211, 2)
        
        # type(df_6sig_8soi_3p_log10)
        # Out[815]: pandas.core.frame.DataFrame
        
        # df_6sig_8soi_3p_log10[:4]
        # Out[816]: 
        #                                                  Tau_all     A_all
        # animal hemisphere penetration stimulus unit_id                    
        # Lucy   r          1           pt       175     -0.244537 -1.196616
        #                                        29      -1.394196 -2.189876
        #                                        324     -2.637647 -0.630374
        #                                        328      0.428404 -0.820666
                                               

plt.scatter( df_6sig_8soi_3p_log10['Tau_all'] , df_6sig_8soi_3p_log10['A_all'] )
# note : since you're already plotting logarithm of values, do not mistakenly put the 'xscale' ehere as 'log' !

plt.savefig( r'/home/azare/groups/PrimNeu/Aryo/analysis/test/3.pdf')


# %%
###########

data_tr = stsc().fit_transform( df_6sig_8soi_3p_log10 )

model = DBSCAN( eps=1 , min_samples=10 )
model
model_fit = model.fit(data_tr )

labels = model_fit.labels_
np.unique(labels)
df_6sig_8soi_3p_log10['lbl'] = labels


# you can also use c= df_6sig_8soi_3p_log10['lbl'] .
plt.scatter(  df_6sig_8soi_3p_log10['Tau_all'] , df_6sig_8soi_3p_log10['A_all'] , c=labels , cmap='magma' )
# note : since you're already plotting logarithm of values, do not mistakenly put the 'xscale' ehere as 'log' !

plt.xlabel('Tau (s)')
plt.ylabel('A (Hz)')
plt.title('clustered data \n' +
          'Tau_A , >= 6 significant responses \n' +
          '211 values'
          )

plt.savefig( r'/home/azare/groups/PrimNeu/Aryo/analysis/stat/ML/1.pdf')


# %%
###########


jp = sns.jointplot(
                    x=df_6sig_8soi_3p_log10['Tau_all'] , 
                    y=df_6sig_8soi_3p_log10['A_all'] , 
                    hue=df_6sig_8soi_3p_log10['lbl'] ,
                    kind='scatter' , 
                    height=17
                    # note : since you're already plotting logarithm of values, do not mistakenly call logarithmic keywords here !
                    )


plt.xlabel( 'Tau_all_2p (log10 values)(s)' , loc='right', fontsize=20 )  #  lin : linear scale
plt.ylabel( 'A_all_2p (log10 values)(Hz)' , loc='top' , fontsize=20)  #  log : logarithmic scale.
plt.tick_params(labelsize=15)

plt.suptitle( 'clustered data \n' +
            'DBSCAN :  eps=1 , min_samples=10 \n' + 
            'Tau_A , >= 6 significant responses \n' +
            '211 values' ,
            x=0.1 , y=0.95 , horizontalalignment='left' , fontsize=15
            )
    
    
plt.savefig( r'/home/azare/groups/PrimNeu/Aryo/analysis/stat/ML/2.pdf')



# %%
###########

# correspondence between the original data & the ML results depends on the 'order' of data & results.
    # ML functions keep the original order of the data when doing transformations (preprocessing) or providing results (labels).
    #  =>  You can use them alongside each other : non-logarithmically transformed data + labels.

jp = sns.jointplot(
                    x=df_6sig_8soi_3p['Tau_all'] , 
                    y=df_6sig_8soi_3p['A_all'] , 
                    hue = labels ,
                    kind='scatter' , 
                    marginal_kws=dict(log_scale=True ) ,
                    height=17
                   )


plt.xlabel( 'Tau_all_2p (s)' , loc='right', fontsize=20 )  #  lin : linear scale
plt.ylabel( 'A_all_2p (Hz)' , loc='top' , fontsize=20)  #  log : logarithmic scale.
plt.tick_params(labelsize=15)

plt.suptitle( 'clustered data \n' +
            'DBSCAN :  eps=1 (on log transformed data) , min_samples=10 \n' + 
            'Tau_A , >= 6 significant responses \n' +
            '211 values' ,
            x=0.1 , y=0.95 , horizontalalignment='left' , fontsize=15
            )
    
    
plt.savefig( r'/home/azare/groups/PrimNeu/Aryo/analysis/stat/ML/3.pdf')


# %%
###########


