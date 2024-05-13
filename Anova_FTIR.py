import pandas as pd
import numpy as np
import os
import umap
import glob
from scipy.interpolate import interp1d
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.multivariate.manova import MANOVA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Function to Interpolate
def read_and_label(folder_path, label):
    """Read spectra files and assign labels."""
    data_list = []
    for i, file_path in enumerate(glob.glob(os.path.join(folder_path, '*.txt'))):
        data = pd.read_csv(file_path, sep=",", header=None, names=['Wavenumber', 'Intensity'])
        data['Label'] = label
        data['Spectrum_ID'] = f"{label}_{i}"  
        data_list.append(data)
    return pd.concat(data_list, ignore_index=True)

def interpolate_spectrum(wavenumbers, intensities, common_wavenumbers):
    """Interpolate spectrum to common wavenumbers."""
    interp_func = interp1d(wavenumbers, intensities, kind='linear', fill_value="extrapolate", bounds_error=False)
    return interp_func(common_wavenumbers)

# Files
base_path = 'F:\\TUD\\DInesh\\Serum_IRC'
class_folders = {'Benign': 0, 'Low': 1, 'Intermediate': 2, 'High': 3}
common_wavenumbers = np.linspace(start=1000, stop=4000, num=3601)

# Interpolate
all_interpolated_data = []
for folder, label in class_folders.items():
    folder_path = os.path.join(base_path, folder)
    class_data = read_and_label(folder_path, label)
    for spectrum_id, group in class_data.groupby('Spectrum_ID'):
        interpolated_intensities = interpolate_spectrum(group['Wavenumber'].values, group['Intensity'].values, common_wavenumbers)
        all_interpolated_data.append(pd.DataFrame({
            'Wavenumber': common_wavenumbers,
            'Intensity': interpolated_intensities,
            'Label': label,
            'Spectrum_ID': spectrum_id
        }))

# Dataframe
final_data = pd.concat(all_interpolated_data, ignore_index=True)
final_pivoted = final_data.pivot_table(index='Spectrum_ID', columns='Wavenumber', values='Intensity').reset_index()
final_pivoted = final_pivoted.join(final_data[['Spectrum_ID', 'Label']].drop_duplicates().set_index('Spectrum_ID'), on='Spectrum_ID')
final_pivoted.to_csv('final_prepared_spectra.tsv', sep='\t', index=False)

# Anova

data = pd.read_csv('final_prepared_spectra.tsv', sep='\t')

# Function to perform ANOVA
def perform_anova(data):
    results = {}
    for col in data.columns.difference(['Spectrum_ID', 'Label']):
        groups = [group[col].dropna() for name, group in data.groupby('Label')]
        stat, p_value = f_oneway(*groups)
        results[col] = {'Statistic': stat, 'P-value': p_value, 'Test Used': 'ANOVA'}
    return pd.DataFrame(results).T

anova_results = perform_anova(data)
anova_results.to_csv('anova_results.tsv', sep='\t', index_label='Wavenumber')

# Plot setup
plt.rcParams.update({
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 18,
    'axes.labelsize': 18,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18
})

sns.set(style="ticks", rc={
    'axes.spines.top': False,
    'axes.spines.right': False
})

results = pd.read_csv('anova_results.tsv', sep='\t')
results['P-value'] = pd.to_numeric(results['P-value'], errors='coerce')
significant = results[(results['P-value'] < 0.05) & results['P-value'].notna()]

plt.figure(figsize=(10, 5))
plt.scatter(results['Wavenumber'], results['P-value'], color='gray', label='Non-significant')
plt.scatter(significant['Wavenumber'], significant['P-value'], color='red', label='Significant')
plt.axhline(0.05, color='blue', linestyle='--', label='Significance Threshold (0.05)')
plt.yscale('log')
plt.xlabel('Wavenumber')
plt.ylabel('P-value')
plt.legend()
plt.show()


# Manova 1

data = pd.read_csv('final_prepared_spectra.tsv', sep='\t')
data.rename(columns={col: 'w_' + col.replace('.', '_') for col in data.columns if 'Label' not in col and 'Spectrum_ID' not in col}, inplace=True)
dependent_variables = ' + '.join([col for col in data.columns if col.startswith('w_')])
formula = f'{dependent_variables} ~ Label'
manova = MANOVA.from_formula(formula, data=data)
result = manova.mv_test()
print(result)

# Manova reduced through PCA

data.rename(columns={col: 'w_' + col.replace('.', '_') for col in data.columns if 'Label' not in col and 'Spectrum_ID' not in col}, inplace=True)
features = data[[col for col in data.columns if col.startswith('w_')]]
labels = data['Label']
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=25)  #25 gives better results
principal_components = pca.fit_transform(features_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, 26)])
principal_df['Label'] = labels
formula = ' + '.join([f'PC{i}' for i in range(1, 26)]) + ' ~ Label'
manova = MANOVA.from_formula(formula, data=principal_df)
result = manova.mv_test()
print(result)


# Post-Anova 

sns.set(style="ticks", rc={
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.labelweight': 'bold',
    'font.size': 18,
    'font.weight': 'bold'
})

data = pd.read_csv('final_prepared_spectra.tsv', sep='\t')
anova_results = pd.read_csv('anova_results.tsv', sep='\t')
data.columns = [str(col) for col in data.columns]
significant_wavenumbers = anova_results[anova_results['P-value'] < 0.05]['Wavenumber'].tolist()
significant_wavenumbers = [str(w) for w in significant_wavenumbers if str(w) in data.columns]
significant_data = data[['Spectrum_ID'] + significant_wavenumbers + ['Label']]
X = significant_data.drop(['Spectrum_ID', 'Label'], axis=1)
y = significant_data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
predictions = rf_classifier.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Top 10 feature importance with PCA

sns.set(style="ticks", rc={
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.labelweight': 'bold',
    'font.size': 18,
    'font.weight': 'bold'
})

data = pd.read_csv('final_prepared_spectra.tsv', sep='\t')
anova_results = pd.read_csv('anova_results.tsv', sep='\t')
significant_wavenumbers = anova_results[anova_results['P-value'] < 0.05]['Wavenumber'].tolist()
significant_data = data[['Spectrum_ID', 'Label'] + [str(w) for w in significant_wavenumbers if str(w) in data.columns]]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(significant_data.drop(['Spectrum_ID', 'Label'], axis=1))
pca = PCA(n_components=25)
principal_components = pca.fit_transform(features_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(25)])
pca_df['Label'] = significant_data['Label']

plt.figure(figsize=(12, 6))
plt.bar(range(1, 26), pca.explained_variance_ratio_, color='grey', align='center')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
# plt.title('PCA Explained Variance Ratio')
plt.show()

# RandomForest for Feature Importances
X = significant_data.drop(['Spectrum_ID', 'Label'], axis=1)
y = significant_data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
predictions = rf_classifier.predict(X_test)

# Top feature importances
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[-10:]  

plt.figure(figsize=(10, 5))
plt.barh(range(len(indices)), importances[indices], color='grey', align='center')
plt.yticks(range(len(indices)), [f"{float(col):.2f}" for col in X_train.columns[indices]], rotation=0)
plt.xlabel('Relative Importance')
plt.show()

# Cumulative variance

# Styling
sns.set(style="ticks", rc={
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.labelweight': 'bold',
    'font.size': 18,
    'font.weight': 'bold'
})

data = pd.read_csv('final_prepared_spectra.tsv', sep='\t')
features = data.drop(['Spectrum_ID', 'Label'], axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=30)
principal_components = pca.fit_transform(features_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 5))
plt.bar(range(1, 31), explained_variance_ratio, alpha=0.6, align='center', label='Individual explained variance', color='green')
plt.step(range(1, 31), cumulative_variance, where='mid', label='Cumulative explained variance', color='blue')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
# plt.title('PCA Explained Variance Ratio')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# Post-Manova

# Styling 
sns.set(style="ticks", rc={
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.labelweight': 'bold',
    'font.size': 18,
    'font.weight': 'bold'
})

data = pd.read_csv('final_prepared_spectra.tsv', sep='\t')
data.rename(columns={col: f"{float(col):.2f}" if col.replace('.', '', 1).isdigit() else col for col in data.columns}, inplace=True)
features = data[[col for col in data.columns if 'Label' not in col and 'Spectrum_ID' not in col]]
labels = data['Label']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=25)
principal_components = pca.fit_transform(features_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(25)])
principal_df['Label'] = labels
formula = ' + '.join(principal_df.columns[:-1]) + ' ~ Label'
manova = MANOVA.from_formula(formula, data=principal_df)
result = manova.mv_test()
print("MANOVA Results:")
print(result)
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(25)], index=features.columns)

'''
Determine which PCs are significant
plot loadings for significant PCs'

'''

# Wilk's Lamda


data = pd.read_csv('final_prepared_spectra.tsv', sep='\t')
scaler = StandardScaler()
features = scaler.fit_transform(data.drop(['Label', 'Spectrum_ID'], axis=1))
pca = PCA(n_components=10)
principal_components = pca.fit_transform(features)
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(10)])
principal_df['Label'] = data['Label']

# Perform MANOVA
formula = ' + '.join([f'PC{i+1}' for i in range(10)]) + ' ~ Label'
manova = MANOVA.from_formula(formula, data=principal_df)
result = manova.mv_test()

manova_results = pd.DataFrame({
    "Wilks' lambda": result.results['Label']['stat'].iloc[0, 0],
    "F Value": result.results['Label']['stat'].iloc[0, 2],
    "Pr > F": result.results['Label']['stat'].iloc[0, 3]
}, index=[f'PC{i+1}' for i in range(10)])

significant_pcs = manova_results[manova_results['Pr > F'] < 0.05]
print("Significant Principal Components based on MANOVA:")
print(significant_pcs)

plt.figure(figsize=(10, 6))
plt.bar(significant_pcs.index, significant_pcs["Wilks' lambda"], color='lightblue')
plt.ylabel("Wilks' lambda")
plt.title("Significant Principal Components from MANOVA")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

'''
Post processing 
'''

sns.set(style="ticks", rc={
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.labelweight': 'bold',
    'font.size': 18,
    'font.weight': 'bold'
})


data = pd.read_csv('final_prepared_spectra.tsv', sep='\t')
features = data.drop(['Spectrum_ID', 'Label'], axis=1)
labels = data['Label']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=50)  
principal_components = pca.fit_transform(features_scaled)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
umap_embedding = reducer.fit_transform(principal_components)

plt.figure(figsize=(12, 10))
scatter = sns.scatterplot(
    x=umap_embedding[:, 0],
    y=umap_embedding[:, 1],
    hue=labels,
    palette='viridis',
    s=100,  
    alpha=0.6  
)
# plt.title('UMAP', fontsize=20)
plt.xlabel('UMAP 1', fontsize=16)
plt.ylabel('UMAP 2', fontsize=16)
scatter.legend(title='Label', title_fontsize='13', fontsize='12', loc='best')
plt.show()

#Pair plot
'''
Can manually choose top PC and make a chart
'''

sns.set(style="ticks", rc={
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.labelweight': 'bold',
    'font.size': 18,
    'font.weight': 'bold'
})

data = pd.read_csv('final_prepared_spectra.tsv', sep='\t')
features = data.drop(['Spectrum_ID', 'Label'], axis=1)
labels = data['Label']
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=3)  
principal_components = pca.fit_transform(features_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
principal_df['Label'] = labels  
palette = {0: 'blue', 1: 'green', 2: 'yellow', 3: 'red'}

pair_plot = sns.pairplot(principal_df, hue='Label', vars=['PC1', 'PC2', 'PC3'], palette=palette,
                         plot_kws={'alpha': 0.6, 's': 80}, diag_kind='kde')
handles = pair_plot._legend_data.values()
labels = pair_plot._legend_data.keys()
pair_plot.fig.legend(handles=handles, labels=labels, loc='upper right', title='Label', title_fontsize='13', fontsize='12')
pair_plot._legend.remove()  
plt.show()



















