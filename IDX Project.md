# Data Understanding


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# Memanggil dataset
data = pd.read_csv('D:/Data Science/Project/.Data/loan_data_2007_2014.csv', sep=",")
```

    C:\Users\Aditya vidiansyah\AppData\Local\Temp\ipykernel_20496\115363725.py:2: DtypeWarning: Columns (20) have mixed types. Specify dtype option on import or set low_memory=False.
      data = pd.read_csv('D:/Data Science/Project/.Data/loan_data_2007_2014.csv', sep=",")
    


```python
# Membaca csv file dan melihat 5 baris pertama
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>...</th>
      <th>total_bal_il</th>
      <th>il_util</th>
      <th>open_rv_12m</th>
      <th>open_rv_24m</th>
      <th>max_bal_bc</th>
      <th>all_util</th>
      <th>total_rev_hi_lim</th>
      <th>inq_fi</th>
      <th>total_cu_tl</th>
      <th>inq_last_12m</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1077501</td>
      <td>1296599</td>
      <td>5000</td>
      <td>5000</td>
      <td>4975.0</td>
      <td>36 months</td>
      <td>10.65</td>
      <td>162.87</td>
      <td>B</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1077430</td>
      <td>1314167</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27</td>
      <td>59.83</td>
      <td>C</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1077175</td>
      <td>1313524</td>
      <td>2400</td>
      <td>2400</td>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96</td>
      <td>84.33</td>
      <td>C</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1076863</td>
      <td>1277178</td>
      <td>10000</td>
      <td>10000</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49</td>
      <td>339.31</td>
      <td>C</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1075358</td>
      <td>1311748</td>
      <td>3000</td>
      <td>3000</td>
      <td>3000.0</td>
      <td>60 months</td>
      <td>12.69</td>
      <td>67.79</td>
      <td>B</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 75 columns</p>
</div>




```python
# Explorasi dataset
print(data.head())
print(data.info())
print(data.describe())
```

       Unnamed: 0       id  member_id  loan_amnt  funded_amnt  funded_amnt_inv  \
    0           0  1077501    1296599       5000         5000           4975.0   
    1           1  1077430    1314167       2500         2500           2500.0   
    2           2  1077175    1313524       2400         2400           2400.0   
    3           3  1076863    1277178      10000        10000          10000.0   
    4           4  1075358    1311748       3000         3000           3000.0   
    
             term  int_rate  installment grade  ... total_bal_il il_util  \
    0   36 months     10.65       162.87     B  ...          NaN     NaN   
    1   60 months     15.27        59.83     C  ...          NaN     NaN   
    2   36 months     15.96        84.33     C  ...          NaN     NaN   
    3   36 months     13.49       339.31     C  ...          NaN     NaN   
    4   60 months     12.69        67.79     B  ...          NaN     NaN   
    
      open_rv_12m open_rv_24m  max_bal_bc all_util total_rev_hi_lim inq_fi  \
    0         NaN         NaN         NaN      NaN              NaN    NaN   
    1         NaN         NaN         NaN      NaN              NaN    NaN   
    2         NaN         NaN         NaN      NaN              NaN    NaN   
    3         NaN         NaN         NaN      NaN              NaN    NaN   
    4         NaN         NaN         NaN      NaN              NaN    NaN   
    
      total_cu_tl inq_last_12m  
    0         NaN          NaN  
    1         NaN          NaN  
    2         NaN          NaN  
    3         NaN          NaN  
    4         NaN          NaN  
    
    [5 rows x 75 columns]
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 466285 entries, 0 to 466284
    Data columns (total 75 columns):
     #   Column                       Non-Null Count   Dtype  
    ---  ------                       --------------   -----  
     0   Unnamed: 0                   466285 non-null  int64  
     1   id                           466285 non-null  int64  
     2   member_id                    466285 non-null  int64  
     3   loan_amnt                    466285 non-null  int64  
     4   funded_amnt                  466285 non-null  int64  
     5   funded_amnt_inv              466285 non-null  float64
     6   term                         466285 non-null  object 
     7   int_rate                     466285 non-null  float64
     8   installment                  466285 non-null  float64
     9   grade                        466285 non-null  object 
     10  sub_grade                    466285 non-null  object 
     11  emp_title                    438697 non-null  object 
     12  emp_length                   445277 non-null  object 
     13  home_ownership               466285 non-null  object 
     14  annual_inc                   466281 non-null  float64
     15  verification_status          466285 non-null  object 
     16  issue_d                      466285 non-null  object 
     17  loan_status                  466285 non-null  object 
     18  pymnt_plan                   466285 non-null  object 
     19  url                          466285 non-null  object 
     20  desc                         125981 non-null  object 
     21  purpose                      466285 non-null  object 
     22  title                        466264 non-null  object 
     23  zip_code                     466285 non-null  object 
     24  addr_state                   466285 non-null  object 
     25  dti                          466285 non-null  float64
     26  delinq_2yrs                  466256 non-null  float64
     27  earliest_cr_line             466256 non-null  object 
     28  inq_last_6mths               466256 non-null  float64
     29  mths_since_last_delinq       215934 non-null  float64
     30  mths_since_last_record       62638 non-null   float64
     31  open_acc                     466256 non-null  float64
     32  pub_rec                      466256 non-null  float64
     33  revol_bal                    466285 non-null  int64  
     34  revol_util                   465945 non-null  float64
     35  total_acc                    466256 non-null  float64
     36  initial_list_status          466285 non-null  object 
     37  out_prncp                    466285 non-null  float64
     38  out_prncp_inv                466285 non-null  float64
     39  total_pymnt                  466285 non-null  float64
     40  total_pymnt_inv              466285 non-null  float64
     41  total_rec_prncp              466285 non-null  float64
     42  total_rec_int                466285 non-null  float64
     43  total_rec_late_fee           466285 non-null  float64
     44  recoveries                   466285 non-null  float64
     45  collection_recovery_fee      466285 non-null  float64
     46  last_pymnt_d                 465909 non-null  object 
     47  last_pymnt_amnt              466285 non-null  float64
     48  next_pymnt_d                 239071 non-null  object 
     49  last_credit_pull_d           466243 non-null  object 
     50  collections_12_mths_ex_med   466140 non-null  float64
     51  mths_since_last_major_derog  98974 non-null   float64
     52  policy_code                  466285 non-null  int64  
     53  application_type             466285 non-null  object 
     54  annual_inc_joint             0 non-null       float64
     55  dti_joint                    0 non-null       float64
     56  verification_status_joint    0 non-null       float64
     57  acc_now_delinq               466256 non-null  float64
     58  tot_coll_amt                 396009 non-null  float64
     59  tot_cur_bal                  396009 non-null  float64
     60  open_acc_6m                  0 non-null       float64
     61  open_il_6m                   0 non-null       float64
     62  open_il_12m                  0 non-null       float64
     63  open_il_24m                  0 non-null       float64
     64  mths_since_rcnt_il           0 non-null       float64
     65  total_bal_il                 0 non-null       float64
     66  il_util                      0 non-null       float64
     67  open_rv_12m                  0 non-null       float64
     68  open_rv_24m                  0 non-null       float64
     69  max_bal_bc                   0 non-null       float64
     70  all_util                     0 non-null       float64
     71  total_rev_hi_lim             396009 non-null  float64
     72  inq_fi                       0 non-null       float64
     73  total_cu_tl                  0 non-null       float64
     74  inq_last_12m                 0 non-null       float64
    dtypes: float64(46), int64(7), object(22)
    memory usage: 266.8+ MB
    None
              Unnamed: 0            id     member_id      loan_amnt  \
    count  466285.000000  4.662850e+05  4.662850e+05  466285.000000   
    mean   233142.000000  1.307973e+07  1.459766e+07   14317.277577   
    std    134605.029472  1.089371e+07  1.168237e+07    8286.509164   
    min         0.000000  5.473400e+04  7.047300e+04     500.000000   
    25%    116571.000000  3.639987e+06  4.379705e+06    8000.000000   
    50%    233142.000000  1.010790e+07  1.194108e+07   12000.000000   
    75%    349713.000000  2.073121e+07  2.300154e+07   20000.000000   
    max    466284.000000  3.809811e+07  4.086083e+07   35000.000000   
    
             funded_amnt  funded_amnt_inv       int_rate    installment  \
    count  466285.000000    466285.000000  466285.000000  466285.000000   
    mean    14291.801044     14222.329888      13.829236     432.061201   
    std      8274.371300      8297.637788       4.357587     243.485550   
    min       500.000000         0.000000       5.420000      15.670000   
    25%      8000.000000      8000.000000      10.990000     256.690000   
    50%     12000.000000     12000.000000      13.660000     379.890000   
    75%     20000.000000     19950.000000      16.490000     566.580000   
    max     35000.000000     35000.000000      26.060000    1409.990000   
    
             annual_inc            dti  ...  total_bal_il  il_util  open_rv_12m  \
    count  4.662810e+05  466285.000000  ...           0.0      0.0          0.0   
    mean   7.327738e+04      17.218758  ...           NaN      NaN          NaN   
    std    5.496357e+04       7.851121  ...           NaN      NaN          NaN   
    min    1.896000e+03       0.000000  ...           NaN      NaN          NaN   
    25%    4.500000e+04      11.360000  ...           NaN      NaN          NaN   
    50%    6.300000e+04      16.870000  ...           NaN      NaN          NaN   
    75%    8.896000e+04      22.780000  ...           NaN      NaN          NaN   
    max    7.500000e+06      39.990000  ...           NaN      NaN          NaN   
    
           open_rv_24m  max_bal_bc  all_util  total_rev_hi_lim  inq_fi  \
    count          0.0         0.0       0.0      3.960090e+05     0.0   
    mean           NaN         NaN       NaN      3.037909e+04     NaN   
    std            NaN         NaN       NaN      3.724713e+04     NaN   
    min            NaN         NaN       NaN      0.000000e+00     NaN   
    25%            NaN         NaN       NaN      1.350000e+04     NaN   
    50%            NaN         NaN       NaN      2.280000e+04     NaN   
    75%            NaN         NaN       NaN      3.790000e+04     NaN   
    max            NaN         NaN       NaN      9.999999e+06     NaN   
    
           total_cu_tl  inq_last_12m  
    count          0.0           0.0  
    mean           NaN           NaN  
    std            NaN           NaN  
    min            NaN           NaN  
    25%            NaN           NaN  
    50%            NaN           NaN  
    75%            NaN           NaN  
    max            NaN           NaN  
    
    [8 rows x 53 columns]
    


```python
# Melihat total baris dan kolom
data.shape
```




    (466285, 75)




```python
# Mengecek missing values
print(data.isnull().sum())
```

    Unnamed: 0               0
    id                       0
    member_id                0
    loan_amnt                0
    funded_amnt              0
                         ...  
    all_util            466285
    total_rev_hi_lim     70276
    inq_fi              466285
    total_cu_tl         466285
    inq_last_12m        466285
    Length: 75, dtype: int64
    

# Exploratory Data Analysis (EDA)


```python
# Korelasi
correlation, _ = np.corrcoef(data['funded_amnt'], data['last_pymnt_amnt'])
print(f"Korelasi antara jumlah pinjaman dan jumlah pembayaran: {correlation[0]:.2f}")

# Melakukan visualisasi data menggunakan grafik dan plot untuk memahami hubungan antar variabel
plt.figure(figsize=(6, 4))
sns.scatterplot(x='funded_amnt', y='last_pymnt_amnt', data=data)
plt.xlabel('Jumlah Pinjaman')
plt.ylabel('Total Pembayaran Terakhir')
plt.title('Jumlah Pinjaman vs Total Pembayaran Terakhir')
plt.show()
```

    Korelasi antara jumlah pinjaman dan jumlah pembayaran: 1.00
    


    
![png](output_8_1.png)
    



```python
# Replace all string values with NaN
data = data.replace('^\s*$', np.nan, regex=True)
data = data.replace('^\s*-\s*$', np.nan, regex=True)
data = data.replace('^\s*[A-Z]\s*$', np.nan, regex=True)
data = data.replace('^\s*[A-Z]\d\s*$', np.nan, regex=True)
data = data.replace('^[-\s]*$', np.nan, regex=True)
data = data.replace('^[A-Z]\s*$', np.nan, regex=True)
data = data.replace('^[A-Z]\d\s*$', np.nan, regex=True)
data = data.replace('^\s*[A-Z]+\s*$', np.nan, regex=True)
data = data.replace(r'[A-Z][a-z]', np.nan, regex=True)
data = data.replace(r'[A-Z][a-z][\W_]*', np.nan, regex=True)
data = data.replace(r'[a-z][\W_]*', np.nan, regex=True)
data = data.replace(r'[A-Z]', np.nan, regex=True)
data = data.replace(r'\d+', np.nan, regex=True)
data = data.replace(r'.', np.nan, regex=True)

# Convert all remaining string values to float
data = data.applymap(lambda x: float(x) if isinstance(x, str) else x)

# Calculate correlation matrix
corr_matrix = data.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
```

    C:\Users\Aditya vidiansyah\AppData\Local\Temp\ipykernel_20496\3307412490.py:18: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      data = data.applymap(lambda x: float(x) if isinstance(x, str) else x)
    C:\Users\Aditya vidiansyah\anaconda3\Lib\site-packages\seaborn\matrix.py:260: FutureWarning: Format strings passed to MaskedConstant are ignored, but in future may error or produce different behavior
      annotation = ("{:" + self.fmt + "}").format(val)
    


    
![png](output_9_1.png)
    



```python
# Statistik deskriptif
#JUMLAH PINJAMAN
mean_loan_amnt = np.mean(data['funded_amnt'])
median_loan_amnt = np.median(data['funded_amnt'])
std_loan_amnt = np.std(data['funded_amnt'])

print(f"Rata-rata jumlah pinjaman: {mean_loan_amnt:.2f}")
print(f"Median jumlah pinjaman: {median_loan_amnt:.2f}")
print(f"Standar deviasi jumlah pinjaman: {std_loan_amnt:.2f}")

# Visualisasi
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.hist(data['funded_amnt'], bins=np.linspace(0, 30000, 30))
plt.title('Histogram jumlah pinjaman')
plt.xlabel('Jumlah pinjaman')
plt.ylabel('Frekuensi')

# JUMLAH PEMBAYARAN
mean_loan_amnt = np.mean(data['last_pymnt_amnt'])
median_loan_amnt = np.median(data['last_pymnt_amnt'])
std_loan_amnt = np.std(data['last_pymnt_amnt'])

print(f"Rata-rata jumlah pembayaran: {mean_loan_amnt:.2f}")
print(f"Median jumlah pembayaran: {median_loan_amnt:.2f}")
print(f"Standar deviasi jumlah pembayaran: {std_loan_amnt:.2f}")

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.hist(data['last_pymnt_amnt'], bins=np.linspace(0, 30000, 30))
plt.title('Histogram jumlah pembayaran')
plt.xlabel('Jumlah Pembayaran')
plt.ylabel('Frekuensi')

plt.tight_layout()
plt.show()
```

    Rata-rata jumlah pinjaman: 14291.80
    Median jumlah pinjaman: 12000.00
    Standar deviasi jumlah pinjaman: 8274.36
    Rata-rata jumlah pembayaran: 3123.91
    Median jumlah pembayaran: 545.96
    Standar deviasi jumlah pembayaran: 5554.73
    


    
![png](output_10_1.png)
    



    
![png](output_10_2.png)
    


# Data Preparation


```python
# Melihat feature apa saja yang memiliki missing value lebih dari 50%
missing_values = pd.Series(data.isnull().sum() / data.shape[0])
missing_values = missing_values[missing_values > 0.50]
missing_values.sort_values(ascending=False)
```




    term                           1.000000
    grade                          1.000000
    application_type               1.000000
    annual_inc_joint               1.000000
    dti_joint                      1.000000
    verification_status_joint      1.000000
    open_acc_6m                    1.000000
    open_il_6m                     1.000000
    open_il_12m                    1.000000
    open_il_24m                    1.000000
    mths_since_rcnt_il             1.000000
    total_bal_il                   1.000000
    il_util                        1.000000
    open_rv_12m                    1.000000
    open_rv_24m                    1.000000
    max_bal_bc                     1.000000
    all_util                       1.000000
    inq_fi                         1.000000
    total_cu_tl                    1.000000
    last_credit_pull_d             1.000000
    next_pymnt_d                   1.000000
    last_pymnt_d                   1.000000
    url                            1.000000
    sub_grade                      1.000000
    emp_title                      1.000000
    emp_length                     1.000000
    home_ownership                 1.000000
    verification_status            1.000000
    issue_d                        1.000000
    loan_status                    1.000000
    pymnt_plan                     1.000000
    desc                           1.000000
    initial_list_status            1.000000
    purpose                        1.000000
    title                          1.000000
    zip_code                       1.000000
    addr_state                     1.000000
    earliest_cr_line               1.000000
    inq_last_12m                   1.000000
    mths_since_last_record         0.865666
    mths_since_last_major_derog    0.787739
    mths_since_last_delinq         0.536906
    dtype: float64




```python
# Drop feature tersebut
data.dropna(thresh = int(data.shape[0]*0.5), axis=1, inplace=True)
```


```python
# Pengecheckan ulang apakah feature tersebut berhasil di drop
missing_values = pd.Series(data.isnull().sum() / data.shape[0])
missing_values = missing_values[missing_values > 0.50]
missing_values.sort_values(ascending=False)
```




    Series([], dtype: float64)




```python
# Melakukan pengecekan data outlier pada variabel funded_amnt
plt.figure(figsize=(10, 5))
plt.boxplot(data['funded_amnt'])
plt.title('Boxplot Jumlah Pinjaman')
plt.xlabel('Jumlah pinjaman')
plt.ylabel('Nilai')
plt.show()

# Melakukan pengecekan data outlier pada variabel last_pymnt_amnt
# Convert the 'last_pymnt_amnt' column to float
data['last_pymnt_amnt'] = pd.to_numeric(data['last_pymnt_amnt'])
# Remove outliers with last_pymnt_amnt > 5000
data_removed_outliers = data[data.last_pymnt_amnt < 1000]
# Plot a boxplot of the 'last_pymnt_amnt' column
plt.figure(figsize=(10, 5))
plt.boxplot(data_removed_outliers['last_pymnt_amnt'])
plt.title('Boxplot Total Pembayaran Terakhir')
plt.xlabel('Total Pembayaran Terakhir')
plt.show()
```


    
![png](output_15_0.png)
    



    
![png](output_15_1.png)
    



```python
from sklearn.preprocessing import MinMaxScaler
# Melakukan scaling pada fitur-fitur numerik funded_amnt
scaler = MinMaxScaler()
scaled_funded_amnt = scaler.fit_transform(data[['funded_amnt']])
# Menampilkan hasil scaling
print(scaled_funded_amnt)

# Melakukan scaling pada fitur-fitur numerik last_pymnt_amnt
scaler = MinMaxScaler()
scaled_last_pymnt_amnt = scaler.fit_transform(data[['last_pymnt_amnt']])
# Menampilkan hasil scaling
print(scaled_last_pymnt_amnt)
```

    [[0.13043478]
     [0.05797101]
     [0.05507246]
     ...
     [0.58550725]
     [0.04347826]
     [0.27536232]]
    [[0.00473638]
     [0.00330238]
     [0.01793625]
     ...
     [0.01419478]
     [0.04141585]
     [0.01014449]]
    


```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Create a new dataframe with scaled features
scaled_data = pd.DataFrame(scaled_funded_amnt, columns=['scaled_funded_amnt'])
scaled_data['scaled_last_pymnt_amnt'] = scaled_last_pymnt_amnt

# Split the data into train and test sets
X = scaled_data.drop('scaled_funded_amnt', axis=1)
y = scaled_data['scaled_funded_amnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the train and test sets
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)
```

    Train set: (373028, 1) (373028,)
    Test set: (93257, 1) (93257,)
    

# Data Modelling


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Memilih model machine learning
# Melakukan pelatihan model pada set pelatihan
# Menyesuaikan parameter model 

# Create a random forest regressor with 100 trees
rf = RandomForestRegressor(n_estimators=100, random_state=42)
# Fit the regressor to the training data
rf.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = rf.predict(X_test)
# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
```

    Mean Squared Error: 0.01905073678994312
    

# Evaluasi Model


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Mengevaluasi kinerja model menggunakan metrik

# Convert continuous values in y to binary classes using a threshold value
threshold = 0.5
y_binary = (y >= threshold).astype(int)
# Split the dataset into training and testing sets
X_train, X_test, y_train_binary, y_test_binary = train_test_split(X, y_binary, test_size=0.2, random_state=42)
# Create a logistic regression model
model = LogisticRegression()
# Train the model on the training data
model.fit(X_train, y_train_binary)
# Make predictions on the testing data
y_pred_prob = model.predict_proba(X_test)[:, 1] # Predict probabilities of positive class
y_pred_labels = (y_pred_prob >= threshold).astype(int) # Convert probabilities to binary labels
# Calculate evaluation metrics
accuracy = accuracy_score(y_test_binary, y_pred_labels)
precision = precision_score(y_test_binary, y_pred_labels)
recall = recall_score(y_test_binary, y_pred_labels)
f1 = f1_score(y_test_binary, y_pred_labels)
# Print the results
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
```

    Accuracy: 0.7340253278574262
    Precision: 0.7608695652173914
    Recall: 0.2080741047974561
    F1 Score: 0.3267831940071653
    


```python
from sklearn.model_selection import cross_val_score

# Memeriksa overfitting atau underfitting model

# Perform 5-fold cross-validation
scores = cross_val_score(rf, X_train, y_train, cv=5)

# Calculate the mean and standard deviation of the scores
mean_score = np.mean(scores)
std_score = np.std(scores)

print("Mean score: {:.2f}".format(mean_score))
print("Standard deviation: {:.2f}".format(std_score))
```

    Mean score: 0.66
    Standard deviation: 0.00
    


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
