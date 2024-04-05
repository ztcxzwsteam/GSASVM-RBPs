import pandas as pd

data = pd.read_csv('./ass_del.csv', header=None, index_col=None)
data2 = pd.read_csv('./ass_del2.csv', header=None, index_col=None)
data3 = pd.read_csv('./m_ss.csv', header=None, index_col=None)
data4 = pd.read_csv('./p_ss.csv', header=None, index_col=None)
data5 = pd.read_csv('./miRNA_seq.csv',  index_col=None)
data6 = pd.read_csv('./protein_seq.csv',  index_col=None)

print(data.shape,  data2.shape, data3.shape, data4.shape)
print(data5.shape, data6.shape)
print('a')