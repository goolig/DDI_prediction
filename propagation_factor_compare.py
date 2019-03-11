import matplotlib.pyplot as plt

from d2d_evaluation import dpi, image_ext

plt.figure(figsize=(5.2, 5))

#retro:
results_test= [(0, 1, 0.799285717747392, 0.3354538683077973), (0.1, 1, 0.8074656096251673, 0.3340720531646644), (0.2, 1, 0.8101879140321108, 0.34089405462517225), (0.3, 1, 0.8098035837784787, 0.3526857853769537), (0.4, 1, 0.8074178459790978, 0.3680466243619263), (0.5, 1, 0.8031187160970885, 0.3870065411740701), (0.6, 1, 0.7961616246052539, 0.41039116224622024), (0.7, 1, 0.784976293337977, 0.4391905094956591), (0.8, 1, 0.7672036272861096, 0.47414562199382115), (0.9, 1, 0.7407700975145761, 0.5157771428693485), (1, 1, 0.706806140259653, 0.5646543812221834)]
results_validation= [(0, 1, 0.6881569628531067, 0.49546834145323154), (0.1, 1, 0.700275100571204, 0.422628558701739), (0.2, 1, 0.7095471419139354, 0.36391571567663683), (0.3, 1, 0.7160268229032603, 0.31771039280674584), (0.4, 1, 0.7197915975671327, 0.2825595756618093), (0.5, 1, 0.7205186984095244, 0.2574651876123106), (0.6, 1, 0.7172162805573298, 0.24201980804097817), (0.7, 1, 0.7084290140619286, 0.23606125461576435), (0.8, 1, 0.692856795570439, 0.2391388337049579), (0.9, 1, 0.6708860497791049, 0.25049939388724457), (1, 1, 0.6463902493989249, 0.269536527441195)]
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
results_test_holdout= [(0, 1, 0.943507197397108, 0.2382833439446425), (0.1, 1, 0.9443338102310108, 0.2361643413580656),
                       (0.2, 1, 0.94728156364545, 0.23053358966604567), (0.3, 1, 0.952479003280903, 0.22135767084532906),
                       (0.4, 1, 0.9595589624989139, 0.2087629695207859), (0.5, 1, 0.9676962663087922, 0.19305526170823292),
                       (0.6, 1, 0.975724417040685, 0.17477174084328853), (0.7, 1, 0.9824560182435906, 0.15478171241362954),
                       (0.8, 1, 0.9871573047564226, 0.13444377144563607), (0.9, 1, 0.9898315670019109, 0.11565889255964337),
                       (1, 1, 0.9909379593038753, 0.10041854116272722)]
results_validation_holdout= [(0, 1, 0.9269204304828732, 0.18214957307184892), (0.1, 1, 0.9281880822339754, 0.18074923512190796),
                             (0.2, 1, 0.9313092761141328, 0.17729769923791233), (0.3, 1, 0.936396147972216, 0.17176029464276238),
                             (0.4, 1, 0.9430985668458266, 0.16425563767545936), (0.5, 1, 0.9506420409420493, 0.15506701571583226),
                             (0.6, 1, 0.9579569238083789, 0.14468613097956537), (0.7, 1, 0.9639837304508906, 0.13390919593449924),
                             (0.8, 1, 0.9681110199451359, 0.12396833984528419), (0.9, 1, 0.9703578340252168, 0.11656593084437065),
                             (1, 1, 0.9710898487640541, 0.11346425532995387)]
ax=plt.subplot(212)
x_test_holdout = [1-a[0] for a in results_test_holdout] #in the paper, alpha is used as complete to one
y_test_holdout = [a[2] for a in results_test_holdout]
x_val_holdout = [1-a[0] for a in results_validation_holdout] #in the paper, alpha is used as complete to one
y_val_holdout = [a[2] for a in results_validation_holdout]
plt.plot(x_val_holdout, y_val_holdout,label='Validation')
plt.plot(x_test_holdout, y_test_holdout,label='Test')
plt.ylabel("AUROC")
plt.title('Propagation factor analysis, holdout')
plt.xlabel("Propagation factor")
plt.locator_params(axis='y', nbins=4)
plt.text(-0.1, 1.05, 'B', size=10, weight='bold',transform=ax.transAxes)
plt.legend()
plt.tight_layout()
plt.savefig(r'results\Propagation_factor_compare' + '.' + image_ext, format=image_ext,dpi=dpi)



plt.show()