import matplotlib.pyplot as plt

from d2d_evaluation import dpi, image_ext

plt.figure(figsize=(5.2, 5))

#retro:
results_test= [(0, 1, 0.7908700782796774, 0.6717437001266313), (0.1, 1, 0.7971976807486771, 0.6737460862202281), (0.2, 1, 0.7991164821150942, 0.6815582621250457), (0.3, 1, 0.7986313245277136, 0.6944082794532642), (0.4, 1, 0.7967242874481746, 0.7119648073811339), (0.5, 1, 0.7934822900509644, 0.7342286686041268), (0.6, 1, 0.7881959293258906, 0.7614303491056122), (0.7, 1, 0.7793071179650808, 0.7938483224459987), (0.8, 1, 0.7645457446686914, 0.8316971052810845), (0.9, 1, 0.7421687852360671, 0.8750537271824392), (1, 1, 0.7130144048533373, 0.9237653160409495)]
results_validation= [(0, 1, 0.6765916966785255, 0.2871166253455981), (0.1, 1, 0.6949816947735008, 0.27859969444801236), (0.2, 1, 0.709301697759718, 0.28263960582768116), (0.3, 1, 0.719385330922132, 0.2938064062285375), (0.4, 1, 0.7247078724718077, 0.30902825025783165), (0.5, 1, 0.7241735367903497, 0.3273399614705026), (0.6, 1, 0.717216171645414, 0.3484656585554634), (0.7, 1, 0.7043403123224479, 0.3721084123912894), (0.8, 1, 0.6868439755998099, 0.39809502323429113), (0.9, 1, 0.6669123377784743, 0.4264684825515989), (1, 1, 0.6472403500825473, 0.45689537971405036)]
ax=plt.subplot(211)
x_test = [1 - a[0] for a in results_test] #in the paper, alpha is used as complete to one
y_test = [a[2] for a in results_test]
x_val = [1- a[0] for a in results_validation] #in the paper, alpha is used as complete to one
y_val = [a[2] for a in results_validation]
plt.plot(x_val, y_val,label='Validation')
plt.plot(x_test, y_test,label='Test')
plt.ylabel("AUROC")
plt.title('Propagation factor analysis, retrospective')
plt.xlabel("Propagation factor")
plt.locator_params(axis='y', nbins=4)
plt.text(-0.1,1.05, 'A', size=10, weight='bold',transform=ax.transAxes)
plt.legend()


#holdout
results_test_holdout= [(0, 1, 0.943507197397108, 0.2382833439446425), (0.1, 1, 0.9443338102310108, 0.2361643413580656), (0.2, 1, 0.94728156364545, 0.23053358966604567), (0.3, 1, 0.952479003280903, 0.22135767084532906), (0.4, 1, 0.9595589624989139, 0.2087629695207859), (0.5, 1, 0.9676962663087922, 0.19305526170823292), (0.6, 1, 0.975724417040685, 0.17477174084328853), (0.7, 1, 0.9824560182435906, 0.15478171241362954), (0.8, 1, 0.9871573047564226, 0.13444377144563607), (0.9, 1, 0.9898315670019109, 0.11565889255964337), (1, 1, 0.9909379593038753, 0.10041854116272722)]
results_validation_holdout= [(0, 1, 0.9269204304828732, 0.18214957307184892), (0.1, 1, 0.9281880822339754, 0.18074923512190796), (0.2, 1, 0.9313092761141328, 0.17729769923791233), (0.3, 1, 0.936396147972216, 0.17176029464276238), (0.4, 1, 0.9430985668458266, 0.16425563767545936), (0.5, 1, 0.9506420409420493, 0.15506701571583226), (0.6, 1, 0.9579569238083789, 0.14468613097956537), (0.7, 1, 0.9639837304508906, 0.13390919593449924), (0.8, 1, 0.9681110199451359, 0.12396833984528419), (0.9, 1, 0.9703578340252168, 0.11656593084437065), (1, 1, 0.9710898487640541, 0.11346425532995387)]
ax=plt.subplot(212)
x_test_holdout = [1-a[0] for a in results_test_holdout] #in the paper, alpha is used as complete to one
y_test_holdout = [a[2] for a in results_test_holdout]
x_val_holdout = [1-a[0] for a in results_validation_holdout] #in the paper, alpha is used as complete to one
y_val_holdout = [a[2] for a in results_validation_holdout]
plt.plot(x_val_holdout, y_val_holdout,label='Validation')
plt.plot(x_test_holdout, y_test_holdout,label='Test')
plt.ylabel("AUROC")
plt.title('Propagation factor analysis, hold-out')
plt.xlabel("Propagation factor")
plt.locator_params(axis='y', nbins=4)
plt.text(-0.1, 1.05, 'B', size=10, weight='bold',transform=ax.transAxes)
plt.legend()
plt.tight_layout()
plt.savefig('Propagation_factor_compare' + '.' + image_ext, format=image_ext,dpi=dpi)



plt.show()