import scipy
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

from d2d_evaluation import image_ext, dpi

structural_sim_file = r'deepddi\data\drug_similarity.csv'
ddi_sim_file = r'pickles\predictions.csv'

structural_sim = pd.read_csv(structural_sim_file,index_col=0)
ddi_sim = pd.read_csv(ddi_sim_file,index_col=0)

drugs_struct = set(structural_sim.index)
drugs_ddi = set(ddi_sim.index)
intersecting_drugs =drugs_ddi & drugs_struct
print(f'strcut len: {len(drugs_struct)}, ddi len: {len(drugs_ddi)}, intersection len: {len(intersecting_drugs)}')

structural_sim.drop(drugs_struct - intersecting_drugs, axis=1,inplace=True)
ddi_sim.drop(drugs_ddi - intersecting_drugs, axis=1,inplace=True)

structural_sim.drop(drugs_struct - intersecting_drugs,inplace=True)
ddi_sim.drop(drugs_ddi - intersecting_drugs,inplace=True)

intersecting_drugs = list(intersecting_drugs)
comparision_list = []
for i1 in range(len(intersecting_drugs)):
    for i2 in range(i1+1,len(intersecting_drugs)):
        d1,d2 = intersecting_drugs[i1],intersecting_drugs[i2]
        comparision_list.append((ddi_sim[d1][d2],structural_sim[d1][d2] ))

comparision =pd.DataFrame(comparision_list,columns=['ddi_sim','structural_sim'])


X = comparision[comparision.columns[0]]
y = comparision[comparision.columns[1]]


X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
pearson = scipy.stats.pearsonr(X,y)
print(f'pearson: {pearson[0]}, p-val: {pearson[1]}')

#normalize predictions:
X = (X-X.min())
X /= X.max()
y = (y-y.min())
y/= y.max()


xmin = X.min()
xmax = X.max()
ymin = y.min()
ymax = y.max()


# fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
# ax = axs[0]
# hb = ax.hexbin(X, y, gridsize=50, cmap='inferno')
# ax.axis([xmin, xmax, ymin, ymax])
# ax.set_title("Hexagon binning")
# cb = fig.colorbar(hb, ax=ax)
# cb.set_label('counts')
fig, axs = plt.subplots(ncols=1)
ax = axs
import matplotlib as mpl
hb = ax.hexbin(X, y, gridsize=30,cmap='plasma',bins='log')#,  #,,bins='log'
ax.axis([xmin, xmax, ymin, ymax])
ax.set_title("Correlation: AMF prediction vs Structural similarity")
ax.set_xlabel('AMF prediction')
ax.set_ylabel('Structural similarity')
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Counts (log)')
# ticks = [str(10**(x-1)) for x in cb.get_ticks()]
# cb.set_ticklabels(ticks)
cb.update_ticks()
plt.savefig('compare_figure' + '.' + image_ext, format=image_ext,dpi=dpi)
plt.show()
