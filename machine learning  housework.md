##   machine learning  housework

### 1.最初的代码实现

```python
# coding = 'utf-8'
import time
import tm
import numpy as np
import pandas as pd


def target_mean_v1(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean', 'count'])
        result[i] = groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')]
    return result

def main():

    y = np.random.randint(2, size=(5000, 1))
    x = np.random.randint(10, size=(5000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])

    start = time.time()
    result_1 = target_mean_v1(data, 'y', 'x')
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()

```

测试时间

![](C:\Users\Administrator\Desktop\ds.png)

## 2.经过改进的代码 target_mean_v2

```python
def target_mean_v2(data: pd.DataFrame, y_name: str, x_name: str) -> np.ndarray:
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result
```

测试时间

![](C:\Users\Administrator\Desktop\wet.png)

## 3.构建setup.py 文件

```python
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

compile_flags = ['-std=c++11',  '-fopenmp']
linker_flags = ['-fopenmp']

module = Extension('tm',
                   ['tm.pyx'],
                   language='c++',
                   include_dirs=[numpy.get_include()], # This helps to create numpy
                   extra_compile_args=compile_flags,
                   extra_link_args=linker_flags)

setup(
    name='tm',
    ext_modules=cythonize(module),
)
```

## 4.创建一个名为tm.pyx的cython文件

```python
# distutils: language=c++
import numpy as np
cimport numpy as cnp
import pandas as pd


def hello():
    print("hello")


def target_mean_v3(data:pd.DataFrame, y_name:str, x_name:str) -> np.ndarray:
  cdef:
    int data_shape = data.shape[0]
    cnp.ndarray[cnp.float64_t] result = np.zeros(data_shape, dtype=np.float64)
    dict value_dict = {}
    dict count_dict = {}
    cnp.ndarray[cnp.int_t] x_val_array = data[x_name].values
    cnp.ndarray[cnp.int_t] y_val_array = data[y_name].values

  for i in range(data_shape):
    data_loc_x = x_val_array[i]
    data_loc_y = y_val_array[i]
    if data_loc_x not in value_dict:
      value_dict[data_loc_x] = data_loc_y
      count_dict[data_loc_x] = 1
    else:
      value_dict[data_loc_x] += data_loc_y
      count_dict[data_loc_x] += 1
  for i in range(data_shape):
    count = count_dict[x_val_array[i]] - 1
    result[i] = (value_dict[x_val_array[i]] - y_val_array[i]) / count

  return result
```

**通过 python setup.py install 和 python setup.py build_ext --inplace 编译成链接，测试时间如下**

![](C:\Users\Administrator\Desktop\rsv_20210513224012.png)

## 5.在10万行观测数据如下

![](C:\Users\Administrator\Desktop\微信截图_20210513225009.png)

