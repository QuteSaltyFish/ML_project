import pandas as pd 
id = []
name = []

id.append(1)
name.append('1123')
test_dict = {'id':id,'name':name}
test_dict_df = pd.DataFrame(test_dict)
print(test_dict_df)
#[2].字典型赋值
test_dict_df = pd.DataFrame(data=test_dict)
print(test_dict_df)