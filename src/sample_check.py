# importing recurired modules
import pandas as pd
import numpy as np 

class SampleCheckData:
    def __init__(self):
        self.__dataset_loc ="DataSets/Data_Dictionary.xlsx"

    def see_head_data(self):
        self.__data = pd.read_excel(self.__dataset_loc)
        return self.__data.head(n=5)

    def see_information(self):
        return self.__data.info()

    def main(self):
        print("Top 5 Rows of a DataSet : \n")
        print(self.see_head_data())
        print("Information of DataSet: \n")
        print(self.see_information ())

if __name__ =="__main__":
    print("="*50)
    print("The University DataSet Cleaning PipleLine : \n")
    obj = SampleCheckData()
    obj.main()
    print("="*50)


