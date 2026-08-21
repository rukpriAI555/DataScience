class Univariate():
    def QuanQual(dataset):
        quan = []
        qual = []
        for columnName in dataset.columns:
            #print(columnName)
            if(dataset[columnName].dtypes=='O'):
                qual.append(columnName)
            else:
                quan.append(columnName)
        return quan, qual