import pandas as pd
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('Repos/pydata-book/ch02/movielens/users.dat', sep='::', header=None,names=unames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('Repos/pydata-book/ch02/movielens/ratings.dat', sep='::', header=None, names=rnames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('Repos/pydata-book/ch02/movielens/movies.dat', sep='::', header=None, names=mnames)

#method .ix inverted index
data.ix[0]

#mean pivot table method
In [341]: mean_ratings = data.pivot_table('rating', rows='title', cols='gender', aggfunc='mean')

#groupby
ratings_by_title = data.groupby('title').size()
active_titles = ratings_by_title.index[ratings_by_title >= 250]

#inverted index
mean_ratings = mean_ratings.ix[active_titles]
mean_ratings

#index results in descending order
top_female_ratings = mean_ratings.sort_index(by='F', ascending=False)
top_female_ratings[:10]

# Measuring rating disagreement. creates 'diff' column and sorts by 'diff'
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_index(by='diff')
sorted_by_diff[:15]

#reverse order of the, take first 15 rows
sorted_by_diff[::-1][:15]


# Standard deviation of rating grouped by title
rating_std_by_title = data.groupby('title')['rating'].std()
# Filter down to active_titles
rating_std_by_title = rating_std_by_title.ix[active_titles]
# Order Series by value in descending order
rating_std_by_title.order(ascending=False)[:10]

#us baby names
!head -n 10 Downloads/names/yob1880.txt
import pandas as pd
names1880 = pd.read_csv('Downloads/names/yob1880.txt', names=['name', 'sex', 'births'])
names1880

names1880.groupby('sex').births.sum()


# 2010 is the last available year right now

years = range(1880, 2011)

pieces = []
columns = ['name', 'sex', 'births']

for year in years:
	path = 'Downloads/names/yob%d.txt' % year
	frame = pd.read_csv(path, names=columns)
	frame['year'] = year
	pieces.append(frame)
	# Concatenate everything into a single DataFrame
	names = pd.concat(pieces, ignore_index=True)

#agregating data
total_births = names.pivot_table('births', rows='year', cols='sex', aggfunc=sum)

total_births.tail()

total_births.plot(title='Total births by sex and year')

#insert a column prop

def add_prop(group):
	# Integer division floors
	births = group.births.astype(float)
	group['prop'] = births / births.sum()
	return group
	names = names.groupby(['year', 'sex']).apply(add_prop)



np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1)

#yet another group operation

def get_top1000(group):
	return group.sort_index(by='births', ascending=False)[:1000]
grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)

#smaller results

top1000

#Analyzing naming trends

boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']

#pivot table
total_births = top1000.pivot_table('births', rows='year', cols='name', aggfunc=sum)

#dateframe plot method

subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12, 10), grid=False, title="Number of births per year")

#measuring the increase in naming diversity
table = top1000.pivot_table('prop', rows='year', cols='sex', aggfunc=sum)
#increase in name diversity
table.plot(title='Sum of table1000.prop by year and sex', yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))

prop_cumsum = df.sort_index(by='prop', ascending=False).prop.cumsum()

df = boys[boys.year == 1900]

in1900 = df.sort_index(by='prop', ascending=False).prop.cumsum()

in1900.searchsorted(0.5) + 1

#diversity data frame
def get_quantile_count(group, q=0.5):
	group = group.sort_index(by='prop', ascending=False)
	return group.prop.cumsum().searchsorted(q) + 1
diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')

#diversity time series

diversity.plot(title="Number of popular names in top 50%")

#the last letter revolution
#extract last letter from the column

get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'
table = names.pivot_table('births', rows=last_letters, cols=['sex', 'year'], aggfunc=sum)

#select out representative years

subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
subtable.head()
subtable.sum()

letter_prop = subtable / subtable.sum().astype(float)

#plot in matplotlib

import matplotlib.pyplot as plt
￼
#making each column a time series
￼
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female',legend=False)

letter_prop = table / table.sum().astype(float)
dny_ts = letter_prop.ix[['d', 'n', 'y'], 'M'].T
dny_ts.head()

#time series plot
dny_ts.plot()

#names that changed over time
all_names = top1000.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]
lesley_like

#filtered down by group names

filtered = top1000[top1000.name.isin(lesley_like)]
filtered.groupby('name').births.sum()

#lets agregate by sex and year and normalize within a year

table = filtered.pivot_table('births', rows='year', cols='sex', aggfunc='sum')
table = table.div(table.sum(1), axis=0)
table.tail()

#plot of the breakdown by sex over time
table.plot(style={'M': 'k-', 'F': 'k--'})

#chapter3

an_apple = 27
an_example = 42
an<Tab>

an_apple and an_example any

#tab key
b = [1, 2, 3]
b.<Tab>

def add_numbers(a, b):
	"""
	Add two numbers together

	Returns
	-------
	the_sum : type of arguments
	"""
	return a + b

	?
	??
# the  %run  command

thon_script_test.py:

def f(x, y, z):
	return (x + y) / z

a =5
b =6
c = 7.5

result = f(a, b, c)

%run ipython_script_test.py

a = np.random.randn(100, 100)
%timeit np.dot(a, a)
10000 loops, best of 3: 69.1 us per loop


#magic commands

# Logging the Input and Output

%logstart

foo = 'test*'

foo = 'test*'
!ls $foo
test4.py test.py test.xml

#python
%alias ll ls -l

#alias method
%alias test_alias (cd ch08; ls; cd ..)

#directory bookmark system

In [6]: %bookmark db /home/wesm/Dropbox/
#
In [7]: cd db
(bookmark:db) -> /home/wesm/Dropbox/ /home/wesm/Dropbox


#run a debugger
run Repos/pydata-bookch03/ipython_bug.py

%debug

#Timing Code with %time and %timeit

%time method1 = [x for x in strings if x.startswith('foo')]
%timeit [x for x in strings if x[:3] == 'foo']\

# basic profiling %prun %run -p

import numpy as np
from numpy.linalg import eigvals
def run_experiment(niter=100):
	K = 100
	results = []
	for _ in xrange(niter):
		mat = np.random.randn(K, K)
		max_eigenvalue = np.abs(eigvals(mat)).max()
		results.append(max_eigenvalue)
	return results
some_results = run_experiment()
print 'Largest one we saw: %s' % np.max(some_results)

python -m cProfile cprof_example.py
python -m cProfile -s cumulative cprof_example.py

%prun -l 7 -s cumulative run_experiment()

#profiling by function line-by-line

from numpy.random import randn

def add_and_sum(x, y):
	added = x + y
	summed = added.sum(axis=1) return summed

def call_function():
	x = randn(1000, 1000)
	y = randn(1000, 1000)
	return add_and_sum(x, y)

%run prof_mod

x = randn(3000, 3000)
y = randn(3000, 3000)

%prun add_and_sum(x, y)

class Message:
	def __init__(self, msg):
		self.msg = msg


# creat an array function
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2

# Data Types for ndarrays

In [27]: arr1 = np.array([1, 2, 3], dtype=np.float64)
In [28]: arr2 = np.array([1, 2, 3], dtype=np.int32)

In [29]: arr1.dtype
Out[29]: dtype('float64')

In [30]: arr2.dtype
Out[30]: dtype('int32')


arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])

#numbers in numeric form



numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)

numeric_strings.astype(float)

git config --global user.name "Jonatan Rodriguez"

git config --global user.email "jrguezg1@gmail.com"

sudo mv git-credential-osxkeychain `dirname `/usr/libexec/git-core``

empty_uint32 = np.empty(8, dtype='u4')

empty_uint32

#operations between Arrays and Scalars

arr = np.array([[1., 2., 3.], [4., 5., 6.]])

# basic indexing and slicing

In [51]: arr = np.arange(10)
In [52]: arr

Out[52]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

In [53]: arr[5] Out[53]: 5
In [54]: arr[5:8] Out[54]: array([5, 6, 7])
In [55]: arr[5:8] = 12
In [56]: arr

Out[56]: array([ 0, 1, 2, 3, 4, 12, 12, 12, 8, 9])

arr_slice = arr[5:8]

arr_slice[1] = 12345

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]

Out[63]: array([7, 8, 9])


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = randn(7, 4)
names

array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'], dtype='|S4')


#Fancy indexing

arr = np.empty((8,4))

for i in range(8):
	arr[i] = i


arr = np.arange(32).reshape((8, 4))

#tranposing Arrays and swapping axes

arr = np.arange(15).reshape((3, 5))

#matrix computations

arr = np.random.randn(6, 3)
np.dot(arr.T, arr)

#data processing arrays

points = np.arange(-5, 5, 0.01) # 1000 equally spaced points

xs, ys = np.meshgrid(points, points)

import matplotlib.pyplot as plt

In [7]: z = np.sqrt(xs ** 2 + ys ** 2)
z

plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
Out[137]: <matplotlib.colorbar.Colorbar instance at 0x4e46d40>

plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
Out[138]: <matplotlib.text.Text at 0x4565790>

#Expressing Conditional Logic as Array Operations

x if condi tion else y

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])

cond = np.array([True, False, True, True, False])

In [26]: result = [(x if c else y)
   ....: for x, y, c in zip(xarr, yarr, cond)]




In [27]: result
Out[27]: [1.1000000000000001, 2.2000000000000002, 1.3, 1.3999999999999999, 2.5]

In [28]: result = np.where(cond, xarr, yarr)

In [29]: result
Out[29]: array([ 1.1,  2.2,  1.3,  1.4,  2.5])

In [30]: arr = randn(4,4)

In [31]: arr
Out[31]:
array([[ 0.41874258,  0.48486755, -1.70226337,  0.16275618],
       [-0.24167952,  0.01220327,  0.725407  ,  1.37563996],
       [ 0.79233435, -1.08395748, -0.84864348,  0.62794599],
       [-0.91987313,  1.89445335,  0.11862593,  0.10668056]])

In [32]: np.where(arr > 0, 2, -2)
Out[32]:
array([[ 2,  2, -2,  2],
       [-2,  2,  2,  2],
       [ 2, -2, -2,  2],
       [-2,  2,  2,  2]])

In [33]: np.where(arr > 0, 2, arr) #set only positive values to 2
Out[33]:
array([[ 2.        ,  2.        , -1.70226337,  2.        ],
       [-0.24167952,  2.        ,  2.        ,  2.        ],
       [ 2.        , -1.08395748, -0.84864348,  2.        ],
       [-0.91987313,  2.        ,  2.        ,  2.        ]])

result = []
for i in range(n):
  if cond1[i] and cond2[i]:
  	result.append(0)
  elif cond1[i]:
  	result.append(1)
  elif cond2[i]:
  	result.append(2)
else:
	result.append(3)
￼￼

#Data Processing Using Arrays


np.where(cond1 & cond2, 0, np.where(cond1, 1,
np.where(cond2, 2, 3)))

#Mathematical nd statistical methods

In [40]: arr = np.random.randn(5, 4) #normally distributed data

In [41]: arr.mean()
Out[41]: 0.12643648648336564

In [42]: np.mean(arr)
Out[42]: 0.12643648648336564

In [43]: arr.sum()
Out[43]: 2.528729729667313

In [40]: arr = np.random.randn(5, 4) #normally distributed data

In [41]: arr.mean()
Out[41]: 0.12643648648336564

In [42]: np.mean(arr)
Out[42]: 0.12643648648336564

In [43]: arr.sum()
Out[43]: 2.528729729667313

In [44]: arr.mean(axis-1)

TypeError: unsupported operand type(s) for -: 'function' and 'int'

In [45]: arr.mean(axis=1)
Out[45]: array([-0.34598076, -0.65001402,  0.55031539,  0.53287924,  0.54498258])

# Mathematical and Statiscal Methods

In [1]: arr = np.random.randn(5, 4) #normally-distributed data

In [2]: arr.mean()
Out[2]: -0.10142844755397398

In [3]: np.mean(arr)
Out[3]: -0.10142844755397398

In [4]: arr.sum()
Out[4]: -2.0285689510794795

In [5]: arr.mean(axis=1)
Out[5]: array([ 0.28027808, -0.05851807,  0.44868888, -0.20171119, -0.97587994])

In [6]: arr.sum(0)
Out[6]: array([-2.29661343,  1.15892748, -1.02068472,  0.12980171])

In [7]: arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

In [8]: arr.cumsum(0)
Out[8]:
array([[ 0,  1,  2],
       [ 3,  5,  7],
       [ 9, 12, 15]])

In [9]: arr.cumprod(1)
Out[9]:
array([[  0,   0,   0],
       [  3,  12,  60],
       [  6,  42, 336]])

# Methods for boolean arrays

n [10]: arr = randn(100)

In [11]: (arr > 0).sum() #number of positive values
Out[11]: 43

In [12]: bools = np.array([False, False, True, False])

In [13]: bools.any()
Out[13]: True

In [16]: bools.all()
Out[16]: False

In [17]: arr = randn(8)

In [18]: arr
Out[18]:
array([ 0.98223588,  0.2161097 , -0.31060441,  0.49456901, -0.25421906,
       -1.95963472,  0.11630277,  0.16163868])

In [19]: arr.sort()

In [20]: arr
Out[20]:
array([-1.95963472, -0.31060441, -0.25421906,  0.11630277,  0.16163868,
        0.2161097 ,  0.49456901,  0.98223588])

In [21]: arr = randn(5, 3)

In [22]: arr
Out[22]:
array([[ 2.96655899, -0.02161218,  0.91198245],
       [ 0.52358395,  0.88596377, -1.29374278],
       [ 0.99126614,  1.84878642, -0.2677843 ],
       [-3.11083741, -1.21499045,  1.18423248],
       [ 1.09382281,  0.41122132, -1.09695888]])

In [23]: arr = randn(8)

In [24]: arr
Out[24]:
array([-1.62737389,  0.43275463,  0.73042864, -1.55579208,  0.27483051,
       -1.12578005, -0.79840802,  0.16850212])

In [25]: arr = randn(5, 3)

In [26]: arr
Out[26]:
array([[-0.94838802,  0.55787217, -0.0105441 ],
       [-0.81158826,  0.05376768, -0.36096404],
       [-0.20358925, -0.44218804, -1.48367411],
       [ 0.48849033, -0.07511989,  0.9329178 ],
       [-2.73319568, -0.71320599, -0.54126072]])

In [27]: arr = randn(5, 3, 6)

In [28]: arr
Out[28]:
array([[[-0.79152827, -1.33160664,  0.90310232,  0.87423416,  1.92274476,
          0.45215774],
        [ 0.36236321, -0.02893653, -2.18937106, -1.24855074, -0.72405438,
         -0.28470789],
        [-0.07992236, -1.03482183, -1.31390579, -0.55487895, -0.79758381,
         -0.69121554]],

       [[ 0.57126146, -0.75287729, -0.14330099,  0.93525401,  0.94484125,
         -0.16856035],
        [ 0.8951046 ,  0.31666688, -0.38332114, -0.99687364, -0.08155014,
          0.52832314],
        [-0.05067398, -0.04578678, -0.6642569 ,  1.01155288,  0.95810103,
         -0.58081079]],

       [[ 0.64171241, -1.06905826, -0.16500841,  0.04305841, -0.36399264,
          0.36822651],
        [ 2.19114162, -0.84132033, -0.06754098,  0.6912863 , -0.34930852,
         -0.27060445],
        [ 1.0724238 ,  1.68162459,  0.78855193, -0.13513414, -0.76400435,
         -0.27223114]],

       [[-0.1989329 ,  0.3721511 , -0.50324904,  1.28096991, -0.36265146,
          0.16142182],
        [-0.66702588, -1.41929438, -1.89866433, -0.45645649,  1.47724571,
          0.93932371],
        [-1.29209896, -0.52767356,  1.08243209,  2.27419236, -0.62530769,
         -0.35078881]],

       [[-0.55465578, -0.6951474 ,  0.37091802, -0.07119193,  1.7489617 ,
         -1.09897234],
        [-0.13890831, -0.34408902, -1.15021078, -2.10961676, -0.55086463,
          0.45304159],
        [ 0.30771625,  1.8283096 ,  0.05627614, -1.36523945, -1.81351581,
          0.06669409]]])

In [29]: arr.sort(1)

In [30]: arr
Out[30]:
array([[[-0.79152827, -1.33160664, -2.18937106, -1.24855074, -0.79758381,
         -0.69121554],
        [-0.07992236, -1.03482183, -1.31390579, -0.55487895, -0.72405438,
         -0.28470789],
        [ 0.36236321, -0.02893653,  0.90310232,  0.87423416,  1.92274476,
          0.45215774]],

       [[-0.05067398, -0.75287729, -0.6642569 , -0.99687364, -0.08155014,
         -0.58081079],
        [ 0.57126146, -0.04578678, -0.38332114,  0.93525401,  0.94484125,
         -0.16856035],
        [ 0.8951046 ,  0.31666688, -0.14330099,  1.01155288,  0.95810103,
          0.52832314]],

       [[ 0.64171241, -1.06905826, -0.16500841, -0.13513414, -0.76400435,
         -0.27223114],
        [ 1.0724238 , -0.84132033, -0.06754098,  0.04305841, -0.36399264,
         -0.27060445],
        [ 2.19114162,  1.68162459,  0.78855193,  0.6912863 , -0.34930852,
          0.36822651]],

       [[-1.29209896, -1.41929438, -1.89866433, -0.45645649, -0.62530769,
         -0.35078881],
        [-0.66702588, -0.52767356, -0.50324904,  1.28096991, -0.36265146,
          0.16142182],
        [-0.1989329 ,  0.3721511 ,  1.08243209,  2.27419236,  1.47724571,
          0.93932371]],

       [[-0.55465578, -0.6951474 , -1.15021078, -2.10961676, -1.81351581,
         -1.09897234],
        [-0.13890831, -0.34408902,  0.05627614, -1.36523945, -0.55086463,
          0.06669409],
        [ 0.30771625,  1.8283096 ,  0.37091802, -0.07119193,  1.7489617 ,
          0.45304159]]])

In [31]: large_arr = randn(1000)

In [32]: large_arr.sort()


In [33]: large_arr[int(0.05 * len(large_arr))] # 5% quantile
Out[33]: -1.552598233233524

# Unique and Other set logic

In [31]: large_arr = randn(1000)

In [32]: large_arr.sort()

In [33]: large_arr[int(0.05 * len(large_arr))] # 5% quantile
Out[33]: -1.552598233233524

In [34]: names = np.array(['Bob', 'Joe', 'Will', 'Joe', 'Joe'])

In [35]: np.unique(names)
Out[35]:
array(['Bob', 'Joe', 'Will'],
      dtype='|S4')

In [36]: sorted(set(names))
Out[36]: ['Bob', 'Joe', 'Will']

In [37]: ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])

In [38]: np.unique(ints)
Out[38]: array([1, 2, 3, 4])

In [40]: sorted(set(names))
Out[40]: ['Bob', 'Joe', 'Will']

In [41]: values = np.array([6, 0, 0, 3, 2, 5, 6])


In [43]: np.in1d(values, [2, 3, 6])
Out[43]: array([ True, False, False,  True,  True, False,  True], dtype=bool)

# Storing Arrays on Disk in binary Format

In [44]: arr = np.arange(10)

In [45]: np.save('some_array', arr)

In [46]: np.oad('some_array.npy')


In [51]: np.savez('array_archive.npz', a=arr, b=arr)

In [52]: arch = np.load('array_archive.npz')

In [53]: arch['b']
Out[53]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

#saving and loading text files

In [191]: !cat array_ex.txt

In [192]: arr = np.loadtxt('array_ex.txt', delimiter=',')
In [193]: arr
Out[193]:
array([[ 0.5801, 0.1867, 1.0407, 1.1344],
[ 0.1942, -0.6369, -0.9387, 0.1241], [-0.1264, 0.2686, -0.6957, 0.0474],
[-1.4844, 0.0042, -0.7442, 0.0055], [ 2.3029, 0.2001, 1.6702, -1.8811],
[-0.1932, 1.0472, 0.4828, 0.9603]])

#linear Algebra

In [56]: x = np.array([[1., 2., 3.], [4., 5., 6.]])

In [58]: y = np.array([[6., 23.], [-1, 7], [8, 9]])

In [59]: x
Out[59]:
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]])

In [60]: y
Out[60]:
array([[  6.,  23.],
       [ -1.,   7.],
       [  8.,   9.]])

In [61]: x.dot(y) # equivalent np.dot(x, y)
Out[61]:
array([[  28.,   64.],
       [  67.,  181.]])

In [62]: np.dot(x, np.ones(3))
Out[62]: array([  6.,  15.])

In [63]: from numpy.linalg import inv, qr

In [64]: X = randn(5, 5)

In [65]: mat = X.T.dot(X)

In [66]: inv(mat)
Out[66]:
array([[ 5.50739926, -0.09954452, -3.88266571,  1.02473459, -2.56862092],
       [-0.09954452,  0.38417847, -0.15801056,  0.19132924,  0.23075963],
       [-3.88266571, -0.15801056,  3.64419754, -1.18148006,  1.69367414],
       [ 1.02473459,  0.19132924, -1.18148006,  0.71173909, -0.21354795],
       [-2.56862092,  0.23075963,  1.69367414, -0.21354795,  1.77731218]])


In [67]: mat.dot(inv(mat))
Out[67]:
array([[  1.00000000e+00,   2.77555756e-17,   0.00000000e+00,
         -1.38777878e-16,   4.44089210e-16],
       [  0.00000000e+00,   1.00000000e+00,   4.44089210e-16,
         -1.38777878e-16,   1.66533454e-16],
       [  8.88178420e-16,   5.55111512e-17,   1.00000000e+00,
          1.11022302e-16,   0.00000000e+00],
       [ -1.77635684e-15,   5.55111512e-17,   4.44089210e-16,
          1.00000000e+00,   4.44089210e-16],
       [  0.00000000e+00,   1.11022302e-16,   8.88178420e-16,
          1.11022302e-16,   1.00000000e+00]])

In [68]: q, r = qr(mat)

In [69]: r
Out[69]:
array([[ 1.86738873,  0.3366732 ,  1.09700116, -0.53893325,  1.84794604],
       [ 0.        ,  3.78237092,  0.47218008, -0.81415496, -1.17907404],
       [ 0.        ,  0.        ,  2.99526635,  5.29346518, -2.42975767],
       [ 0.        ,  0.        ,  0.        ,  1.01322444,  0.40775538],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.2803416 ]])

In [70]: q
Out[70]:
array([[ 0.69271615, -0.01553022,  0.03590528, -0.00908286, -0.72009129],
       [ 0.09343416,  0.95064169, -0.02117903,  0.28795294,  0.06469152],
       [ 0.46060214,  0.08800359,  0.54600098, -0.50649974,  0.47480731],
       [-0.0962967 , -0.16187028,  0.74758842,  0.63407612, -0.05986637],
       [ 0.53850442, -0.24918382, -0.37584137,  0.50833659,  0.49825453]])

#Random Number Generator


In [71]: samples = np.random.normal(size=(4, 4))

In [72]: samples
Out[72]:
array([[ 0.84314887, -1.21018117,  0.5308742 , -1.88065363],
       [-0.33168783,  0.57551299,  0.62729131,  1.03924413],
       [ 0.5166907 , -1.07479986,  0.72052029, -1.60599466],
       [ 1.12087879,  0.55962894, -1.35708189, -0.08780134]])

In [73]: from random import normalvariate

In [74]: N = 1000000

In [76]: %timeit samples = [normalvariate(0, 1) for _ in xrange(N)]
1 loops, best of 3: 1.78 s per loop

In [77]: %timeit np.random.normal(size=N)
10 loops, best of 3: 68.7 ms per loop

#Examples

In [78]: nsteps = 1000

In [80]: draws = np.random.randint(0, 2, size=nsteps)

In [81]: steps = np.where(draws > 0, 1, -1)

In [82]: walk = steps.cumsum()

In [83]: walk.min()
Out[83]: -25

In [84]: walk.max()
Out[84]: 10

In [85]: (np.abs(walk) >= 10).argmax()
Out[85]: 47

#Simulating random walks

n [86]: nwalks = 5000

In [87]: nsteps = 1000

In [88]: draws = np.random.randint(0, 2, size=(nwalks, nsteps)) #0 or 1

In [89]: steps = np.where(draws > 0, 1, -1)

In [90]: walks = steps.cumsum(1)

In [91]: walks
Out[91]:
array([[ 1,  2,  3, ..., 24, 23, 24],
       [ 1,  0, -1, ..., 10,  9, 10],
       [ 1,  0, -1, ..., 30, 31, 30],
       ...,
       [ 1,  0,  1, ...,  6,  5,  4],
       [ 1,  2,  1, ..., -2, -1, -2],
       [-1, -2, -3, ..., -8, -9, -8]])

In [92]: walks.max()
Out[92]: 113

In [93]: walks.min()
Out[93]: -127

In [94]: hits30 = (np.abs(walks) >=30).any(1)

In [95]: hits30
Out[95]: array([ True, False,  True, ..., False, False, False], dtype=bool)

In [96]: hits30.sum() #Number that hit 30 or -30
Out[96]: 3412

In [98]: crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)

In [99]: crossing_times.mean()
Out[99]: 503.79953106682296


#getting started with pandas

In [1]: from pandas import Series, DataFrame
In [2]: import pandas as pd

#series


In [103]: from pandas import Series, DataFrame

In [104]: import pandas as pd

In [105]: obj = Series([4, 7, -5, 3])

In [106]: obj
Out[106]:
0    4
1    7
2   -5
3    3

In [107]: obj.values
Out[107]: array([ 4,  7, -5,  3], dtype=int64)

In [108]: obj.index
Out[108]: Int64Index([0, 1, 2, 3], dtype=int64)

In [109]: obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])

In [110]: obj2
Out[110]:
d    4
b    7
a   -5
c    3

In [111]: obj2.index
Out[111]: Index([d, b, a, c], dtype=object)

n [117]: obj2['d'] = 6

In [118]: obj2[['c', 'a', 'd']]
Out[118]:
c    3
a   -5
d    6

In [119]: obj2
Out[119]:
d    6
b    7
a   -5
c    3

In [120]: obj2[obj2 > 0]
Out[120]:
d    6
b    7
c    3

In [121]: obj2 * 2
Out[121]:
d    12
b    14
a   -10
c     6

In [122]: np.exp(obj2)
Out[122]:
d     403.428793
b    1096.633158
a       0.006738
c      20.085537

In [123]: 'b' in obj2
Out[123]: True

In [124]: 'e' in obj2
Out[124]: False

In [125]: sdata = {'Ohio' : 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}

In [126]: obj3 = Series(sdata)

In [127]: obj3
Out[127]:
Ohio      35000
Oregon    16000
Texas     71000
Utah       5000

In [128]: states = ['California', 'Ohio', 'Texas']

In [129]: obj4 = Series(sdata, index=states)

In [130]: obj4
Out[130]:
California      NaN
Ohio          35000
Texas         71000


In [131]: pd.isnull(obj4)
Out[131]:
California     True
Ohio          False
Texas         False

In [132]: pd.notnull(obj4)
Out[132]:
California    False
Ohio           True
Texas          True

In [133]: obj4.isnull()
Out[133]:
California     True
Ohio          False
Texas         False

In [134]: obj3
Out[134]:
Ohio      35000
Oregon    16000
Texas     71000
Utah       5000

In [135]: obj4
Out[135]:
California      NaN
Ohio          35000
Texas         71000

In [136]: obj3 + obj4
Out[136]:
California       NaN
Ohio           70000
Oregon           NaN
Texas         142000
Utah             NaN

In [137]: obj4.name = 'population'

In [138]: obj4.index.name = 'state'

In [140]: obj4
Out[140]:
state
California      NaN
Ohio          35000
Texas         71000
Name: population

In [141]: obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']

In [142]: obj
Out[142]:
Bob      4
Steve    7
Jeff    -5
Ryan     3

#DataFrame

In [2]: ç

In [3]: import pandas as pd

In [7]: data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 'year': [2000, 2001, 2002, 2001, 2002],
   ...: 'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}

In [8]: frame = DataFrame(data)

In [9]: frame
Out[9]:
   pop   state  year
0  1.5    Ohio  2000
1  1.7    Ohio  2001
2  3.6    Ohio  2002
3  2.4  Nevada  2001
4  2.9  Nevada  2002

In [10]: DataFrame(data,columns=['year', 'state', 'pop'])
Out[10]:
   year   state  pop
0  2000    Ohio  1.5
1  2001    Ohio  1.7
2  2002    Ohio  3.6
3  2001  Nevada  2.4
4  2002  Nevada  2.9


In [13]: frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
   ....: index=['one', 'two', 'three', 'four', 'five'])

In [14]: frame2
Out[14]:
       year   state  pop debt
one    2000    Ohio  1.5  NaN
two    2001    Ohio  1.7  NaN
three  2002    Ohio  3.6  NaN
four   2001  Nevada  2.4  NaN
five   2002  Nevada  2.9  NaN

In [15]: frame2
Out[15]:
       year   state  pop debt
one    2000    Ohio  1.5  NaN
two    2001    Ohio  1.7  NaN
three  2002    Ohio  3.6  NaN
four   2001  Nevada  2.4  NaN
five   2002  Nevada  2.9  NaN

In [16]: frame2.columns
Out[16]: Index([year, state, pop, debt], dtype=object)


In [18]: frame2['state']
Out[18]:
one        Ohio
two        Ohio
three      Ohio
four     Nevada
five     Nevada
Name: state

In [19]: frame2.year
Out[19]:
one      2000
two      2001
three    2002
four     2001
five     2002
Name: year

In [20]: frame2.ix['three']
Out[20]:
year     2002
state    Ohio
pop       3.6
debt      NaN
Name: three

In [21]: frame2['debt'] = 16.5

In [22]: frame2
Out[22]:
       year   state  pop  debt
one    2000    Ohio  1.5  16.5
two    2001    Ohio  1.7  16.5
three  2002    Ohio  3.6  16.5
four   2001  Nevada  2.4  16.5
five   2002  Nevada  2.9  16.5

In [23]: frame2['debt'] = np.arange(5.)

In [24]: frame2
Out[24]:
       year   state  pop  debt
one    2000    Ohio  1.5     0
two    2001    Ohio  1.7     1
three  2002    Ohio  3.6     2
four   2001  Nevada  2.4     3
five   2002  Nevada  2.9     4

In [25]: val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])

In [26]: frame2['debt'] = val

In [27]: frame2
Out[27]:
       year   state  pop  debt
one    2000    Ohio  1.5   NaN
two    2001    Ohio  1.7  -1.2
three  2002    Ohio  3.6   NaN
four   2001  Nevada  2.4  -1.5
five   2002  Nevada  2.9  -1.7

n [28]: frame2['eastern'] = frame2.state == 'Ohio'

In [29]: frame2
Out[29]:
       year   state  pop  debt eastern
one    2000    Ohio  1.5   NaN    True
two    2001    Ohio  1.7  -1.2    True
three  2002    Ohio  3.6   NaN    True
four   2001  Nevada  2.4  -1.5   False
five   2002  Nevada  2.9  -1.7   False

#Index objects

In [3]: from pandas import Series, DataFrame

In [4]: import pandas as pd

In [5]: obj = Series(range(3), index=['a', 'b', 'c'])

In [6]: index = obj.index

In [7]: index
Out[7]: Index([a, b, c], dtype=object)

In [8]: index[1:]
Out[8]: Index([b, c], dtype=object)

In [9]: index[1]
Out[9]: 'b'

In [10]: index[1] = 'd'
---------------------------------------------------------------------------
Exception                                 Traceback (most recent call last)
/Users/Makindo/<ipython-input-10-676fdeb26a68> in <module>()
----> 1 index[1] = 'd'

/Library/Python/2.7/site-packages/pandas-0.10.1-py2.7-macosx-10.8-intel.egg/pandas/core/index.pyc in __setitem__(self, key, value)
    350
    351     def __setitem__(self, key, value):
--> 352         raise Exception(str(self.__class__) + ' object is immutable')
    353
    354     def __getitem__(self, key):

Exception: <class 'pandas.core.index.Index'> object is immutable

##############################
#Index objects are immutable #
##############################

In [14]: index = pd.Index(np.arange(3))

In [15]: obj2 = Series([1.5, -2.5, 0], index=index)

In [16]: obj2.index is index
Out[16]: True


#Essential Functionality

In [17]: obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])

In [18]: obj
Out[18]:
d    4.5
b    7.2
a   -5.3
c    3.6

In [19]: obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])

In [20]: obj2
Out[20]:
a   -5.3
b    7.2
c    3.6
d    4.5
e    NaN

In [21]: obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)
Out[21]:
a   -5.3
b    7.2
c    3.6
d    4.5
e    0.0

In [22]: obj3 - Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/Users/Makindo/<ipython-input-22-52ed86d61b38> in <module>()
----> 1 obj3 - Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])

NameError: name 'obj3' is not defined

In [23]: obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])

In [24]: obj3.reindex(range(6), method= 'ffill')
Out[24]:
0      blue
1      blue
2    purple
3    purple
4    yellow
5    yellow

In [27]: frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'],
   ....: columns=['Ohio', 'Texas', 'California'])

In [28]: frame
Out[28]:
   Ohio  Texas  California
a     0      1           2
c     3      4           5
d     6      7           8

In [29]: frame2 = frame.reindex(['a', 'b', 'c', 'd'])

In [30]: frame2
Out[30]:
   Ohio  Texas  California
a     0      1           2
b   NaN    NaN         NaN
c     3      4           5
d     6      7           8

In [37]: states = ['California', 'Ohio', 'Texas']

In [38]: frame.reindex(index=['a', 'b', 'c', 'd'], method='ffill',
   ....: columns=states)
Out[38]:
   California  Ohio  Texas
a           2     0      1
b           2     0      1
c           5     3      4
d           8     6      7

#dropping entries from an axis

In [40]: obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])

In [41]: new_obj = obj.drop('c')

In [42]: new
new_figure_manager  new_obj             newaxis             newbuffer

In [42]: new_obj
Out[42]:
a    0
b    1
d    3
e    4

In [43]: obj.drop(['d', 'c'])
Out[43]:
a    0
b    1
e    4

In [46]: data = DataFrame(np.arange(16).reshape((4, 4)),
   ....: index=['Ohio', 'Colorado', 'Utah', 'New York'],
   ....: columns=['one', 'two', 'three', 'four'])

In [47]: data.drop(['Colorado', 'Ohio'])
Out[47]:
          one  two  three  four
Utah        8    9     10    11
New York   12   13     14    15

Out[48]:
          one  three  four
Ohio        0      2     3
Colorado    4      6     7
Utah        8     10    11
New York   12     14    15

In [49]: data.drop(['two', 'four'], axis=1)
Out[49]:
          one  three
Ohio        0      2
Colorado    4      6
Utah        8     10
New York   12     14

#indexing, selection and filtering

In [50]: obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])

In [51]: obj['b']
Out[51]: 1.0

In [52]: obj[1]
Out[52]: 1.0

In [53]: obj[2:4]
Out[53]:
c    2
d    3

In [54]: obj[['b', 'a', 'd']]
Out[54]:
b    1
a    0
d    3

In [55]: obj[[1, 3]]
Out[55]:
b    1
d    3

In [56]: obj[obj < 2]
Out[56]:
a    0
b    1

In [57]: obj['b';'c']
  File "<ipython-input-57-0b81b57dca3e>", line 1
    obj['b';'c']
           ^
SyntaxError: invalid syntax


In [58]: obj['b':'c']
Out[58]:
b    1
c    2

In [59]: obj['b':'c'] = 5

In [60]: obj
Out[60]:
a    0
b    5
c    5
d    3


In [64]: data
Out[64]:
          one  two  three  four
Ohio        0    1      2     3
Colorado    4    5      6     7
Utah        8    9     10    11
New York   12   13     14    15

In [65]: data < 5
Out[65]:
            one    two  three   four
Ohio       True   True   True   True
Colorado   True  False  False  False
Utah      False  False  False  False
New York  False  False  False  False

In [66]: data[data < 5] = 0

In [67]: data
Out[67]:
          one  two  three  four
Ohio        0    0      0     0
Colorado    0    5      6     7
Utah        8    9     10    11
New York   12   13     14    15


In [68]: data.ix['colorado', ['two', 'three']]
---------------------------------------------------------------------------

In [69]: data.ix['Colorado', ['two', 'three']]
Out[69]:
two      5
three    6
Name: Colorado

In [70]: data.ix[['Colorado', 'Utah'], [3, 0, 1]]
Out[70]:
          four  one  two
Colorado     7    0    5
Utah        11    8    9

In [71]: data.ix[2]
Out[71]:
one       8
two       9
three    10
four     11
Name: Utah

In [72]: data.ix[:'Utah', 'two']
Out[72]:
Ohio        0
Colorado    5
Utah        9
Name: two

In [73]: data.ix[data.three > 5, :3]
Out[73]:
          one  two  three
Colorado    0    5      6
Utah        8    9     10
New York   12   13     14

# arithmetic and data alignment


In [74]: s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])

In [75]: s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])

In [76]: s1
Out[76]:
a    7.3
c   -2.5
d    3.4
e    1.5

In [77]: s2
Out[77]:
a   -2.1
c    3.6
e   -1.5
f    4.0
g    3.1

In [78]: s1 + s2
Out[78]:
a    5.2
c    1.1
d    NaN
e    0.0
f    NaN
g    NaN


In [80]: df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
   ....: index=['Ohio', 'Texas', 'Colorado'])

In [81]: df2 = DataFrame(np.arange(12.).reshape((4,3)), columns=list('bde'),
   ....: index=['Utah', 'Ohio', 'Texas', 'Oregon'])

In [82]: df1
Out[82]:
          b  c  d
Ohio      0  1  2
Texas     3  4  5
Colorado  6  7  8

In [83]: df2
Out[83]:
        b   d   e
Utah    0   1   2
Ohio    3   4   5
Texas   6   7   8
Oregon  9  10  11

In [84]: df1 + df2
Out[84]:
           b   c   d   e
Colorado NaN NaN NaN NaN
Ohio       3 NaN   6 NaN
Oregon   NaN NaN NaN NaN
Texas      9 NaN  12 NaN
Utah     NaN NaN NaN NaN

# arithmetic methods with fill values

In [85]: DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
Out[85]:
   a  b   c   d
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11

In [86]: df1
Out[86]:
          b  c  d
Ohio      0  1  2
Texas     3  4  5
Colorado  6  7  8

In [87]: In [137]: df2
Out[87]:
        b   d   e
Utah    0   1   2
Ohio    3   4   5
Texas   6   7   8
Oregon  9  10  11


In [89]: df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))

In [90]: df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))

In [91]: df1
Out[91]:
   a  b   c   d
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11

In [92]: df2
Out[92]:
    a   b   c   d   e
0   0   1   2   3   4
1   5   6   7   8   9
2  10  11  12  13  14
3  15  16  17  18  19

In [93]: df1 + df2
Out[93]:
    a   b   c   d   e
0   0   2   4   6 NaN
1   9  11  13  15 NaN
2  18  20  22  24 NaN
3 NaN NaN NaN NaN NaN

In [94]: df1.add(df2, fill_value=0)
Out[94]:
    a   b   c   d   e
0   0   2   4   6   4
1   9  11  13  15   9
2  18  20  22  24  14
3  15  16  17  18  19

In [95]: df1.reindex(columns=df2.columns, fill_value=0)
Out[95]:
   a  b   c   d  e
0  0  1   2   3  0
1  4  5   6   7  0
2  8  9  10  11  0

# operations between dataframe and series


In [11]: arr = np.arange(12.).reshape((3, 4))

In [12]: arr
Out[12]:
array([[  0.,   1.,   2.,   3.],
       [  4.,   5.,   6.,   7.],
       [  8.,   9.,  10.,  11.]])

In [13]: arr[0]
Out[13]: array([ 0.,  1.,  2.,  3.])

In [14]: arr - arr[0]
Out[14]:
array([[ 0.,  0.,  0.,  0.],
       [ 4.,  4.,  4.,  4.],
       [ 8.,  8.,  8.,  8.]])

In [15]: frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
   ....: index=['Utah', 'Ohio', 'Texas', 'Oregon'])

In [16]: series = frame.ix[0]

In [17]: frame
Out[17]:
        b   d   e
Utah    0   1   2
Ohio    3   4   5
Texas   6   7   8
Oregon  9  10  11

In [18]: series
Out[18]:
b    0
d    1
e    2
Name: Utah

In [19]: frame - series
Out[19]:
        b  d  e
Utah    0  0  0
Ohio    3  3  3
Texas   6  6  6
Oregon  9  9  9

In [20]: series2 = Series(range(3), index=['b', 'e', 'f'])

In [21]: frame + series2
Out[21]:
        b   d   e   f
Utah    0 NaN   3 NaN
Ohio    3 NaN   6 NaN
Texas   6 NaN   9 NaN
Oregon  9 NaN  12 NaN

In [22]: series3 = frame['d']

In [23]: frame
Out[23]:
        b   d   e
Utah    0   1   2
Ohio    3   4   5
Texas   6   7   8
Oregon  9  10  11

In [24]: series3
Out[24]:
Utah       1
Ohio       4
Texas      7
Oregon    10
Name: d

In [25]: frame.sub(series3, axis=0)
Out[25]:
        b  d  e
Utah   -1  0  1
Ohio   -1  0  1
Texas  -1  0  1
Oregon -1  0  1

#Function application and mapping
#NumPy ufuncs (element-wise array methods) work fine with panda objects:

In [5]: from pandas import Series, DataFrame

In [6]: import pandas as pd

In [7]: frame = DataFrame(np.random.randn(4, 3), columns=list('bde'),
   ...: index=['Utah', 'Ohio', 'Texas', 'Oregon'])

In [8]: frame
Out[8]:
               b         d         e
Utah   -1.021380  1.039845  1.620518
Ohio    0.946591  0.907383 -1.636917
Texas   1.341285 -0.920366  1.744208
Oregon -0.294026 -0.809490 -1.044885

In [9]: np.abs(frame)
Out[9]:
               b         d         e
Utah    1.021380  1.039845  1.620518
Ohio    0.946591  0.907383  1.636917
Texas   1.341285  0.920366  1.744208
Oregon  0.294026  0.809490  1.044885

#Another frequent operation in 1d arrays to each column or row dataframe method -> apply does that.

In [12]: f = lambda x: x.max() - x.min()

In [13]: frame.apply(f)
Out[13]:
b    2.362665
d    1.960211
e    3.381126

In [14]: frame.apply(f, axis=1)
Out[14]:
Utah      2.641898
Ohio      2.583508
Texas     2.664574
Oregon    0.750859

In [15]: def f(x):
   ....:     return Series([x.min(), x.max()], index=['min', 'max'])
   ....:

In [16]: frame.apply(f)
/Library/Python/2.7/site-packages/pandas-0.10.1-py2.7-macosx-10.8-intel.egg/pandas/core/frame.py:3576: FutureWarning: rename with inplace=True  will return None from pandas 0.11 onward
  " from pandas 0.11 onward", FutureWarning)
Out[16]:
            b         d         e
min -1.021380 -0.920366 -1.636917
max  1.341285  1.039845  1.744208

In [17]: format = lambda x: '%2f' % x

In [18]: frame.applymap(format)
Out[18]:
                b          d          e
Utah    -1.021380   1.039845   1.620518
Ohio     0.946591   0.907383  -1.636917
Texas    1.341285  -0.920366   1.744208
Oregon  -0.294026  -0.809490  -1.044885

In [19]: frame['e'].map(format)
Out[19]:
Utah       1.620518
Ohio      -1.636917
Texas      1.744208
Oregon    -1.044885
Name: e

In [20]: obj = Series(range(4), index=['d', 'a', 'b', 'c'])

In [21]: obj.sort_index()
Out[21]:
a    1
b    2
c    3
d    0

#sorting and ranking

In [2]: from pandas import Series, DataFrame

In [3]: import pandas as pd 

In [4]: ibj = Series(range(4), index=['d', 'a', 'b', 'c'])

In [5]: obj.sort_index()
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/Users/Makindo/<ipython-input-5-733170f67016> in <module>()
----> 1 obj.sort_index()

NameError: name 'obj' is not defined

In [6]: obj = Series(range(4), index=['d', 'a', 'b', 'c'])

In [7]: ibj.sort_index()
Out[7]: 
a    1
b    2
c    3
d    0

In [8]: obj.sort_index()
Out[8]: 
a    1
b    2
c    3
d    0

n [9]: frame = Dataframe(np.arange(8).reshape((2, 4)), index=['three', 'one'],
   ...: columns=['d', 'a', 'b', 'c'])
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/Users/Makindo/<ipython-input-9-e389ed947846> in <module>()
----> 1 frame = Dataframe(np.arange(8).reshape((2, 4)), index=['three', 'one'],
      2 columns=['d', 'a', 'b', 'c'])

NameError: name 'Dataframe' is not defined

In [10]: frame = DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'],   ....: columns=['d', 'a', 'b', 'c'])

In [11]: frame.sort_index()
Out[11]: 
       d  a  b  c
one    4  5  6  7
three  0  1  2  3

In [12]: frame.sort_index(axis=1)
Out[12]: 
       a  b  c  d
three  1  2  3  0
one    5  6  7  4

In [13]: frame.sort_index(axis=1, ascending=False)
Out[13]: 
       d  c  b  a
three  0  3  2  1
one    4  7  6  5

In [14]: obj = Series([4, 7, -3, 2)]
  File "<ipython-input-14-0f3bc1f7c801>", line 1
    obj = Series([4, 7, -3, 2)]
                             ^
SyntaxError: invalid syntax


In [15]: obj = Series([4, 7, -3, 2])

In [16]: obj.order
Out[16]: 
<bound method Series.order of 0    4
1    7
2   -3
3    2>

In [17]: obj = Series([4, np.nan, 7, np.nan, -3, 2])

In [18]: obj.order()
Out[18]: 
4    -3
5     2
0     4
2     7
1   NaN
3   NaN

In [19]: frame = DateFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0,1]})
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/Users/Makindo/<ipython-input-19-557550abd36a> in <module>()
----> 1 frame = DateFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0,1]})

NameError: name 'DateFrame' is not defined

In [20]: frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})

In [21]: frame
Out[21]: 
   a  b
0  0  4
1  1  7
2  0 -3
3  1  2

In [22]: frame.sort_index(by='b')
Out[22]: 
   a  b
2  0 -3
3  1  2
0  0  4
1  1  7

In [23]: frame.sort_index(by=['a', 'b'])
Out[23]: 
   a  b
2  0 -3
0  0  4
3  1  2
1  1  7

In [24]: obj = Series([7, -5, 7, 4, 2, 0, 4])

In [25]: obj.rank()
Out[25]: 
0    6.5
1    1.0
2    6.5
3    4.5
4    3.0
5    2.0
6    4.5

In [26]: obj.rank(method='first')
Out[26]: 
0    6
1    1
2    7
3    4
4    3
5    2
6    5

In [27]: obj.rank(ascending=False, method='max')
Out[27]: 
0    2
1    7
2    2
3    4
4    5
5    6
6    4

In [28]: frame = Dataframe({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1], 
   ....: 'c' : [-2, 5, 8, -2.5]})
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/Users/Makindo/<ipython-input-28-985cc0fdda71> in <module>()
----> 1 frame = Dataframe({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1], 
      2 'c' : [-2, 5, 8, -2.5]})

NameError: name 'Dataframe' is not defined

In [29]: frame = DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1],
   ....: 'c': [-2, 5, 8, -2.5]})

In [30]: frame
Out[30]: 
   a    b    c
0  0  4.3 -2.0
1  1  7.0  5.0
2  0 -3.0  8.0
3  1  2.0 -2.5

In [31]: frame.rank(axis=1)
Out[31]: 
   a  b  c
0  2  3  1
1  1  3  2
2  2  1  3
3  2  3  1

In [32]: 

# axes indexes with duplicate values 

In [1]: from pandas import Series, DataFrame

In [2]: import pandas as pd 

In [3]: obj = Series(range(5), index=['a', 'a', 'b', 'c'])
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/Users/Makindo/<ipython-input-3-3475237754ee> in <module>()
----> 1 obj = Series(range(5), index=['a', 'a', 'b', 'c'])

/Library/Python/2.7/site-packages/pandas-0.10.1-py2.7-macosx-10.8-intel.egg/pandas/core/series.pyc in __new__(cls, data, index, dtype, name, copy)
    385         else:
    386             subarr = subarr.view(Series)
--> 387         subarr.index = index
    388         subarr.name = name
    389 

/Library/Python/2.7/site-packages/pandas-0.10.1-py2.7-macosx-10.8-intel.egg/pandas/lib.so in pandas.lib.SeriesIndex.__set__ (pandas/lib.c:27864)()

AssertionError: Index length did not match values

In [4]: obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])

In [5]: obj
Out[5]: 
a    0
a    1
b    2
b    3
c    4

In [6]: ojb.index.is_unique
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/Users/Makindo/<ipython-input-6-85887cad4995> in <module>()
----> 1 ojb.index.is_unique

NameError: name 'ojb' is not defined

In [7]: obj.index.is_unique
Out[7]: False

In [8]: obj['a']
Out[8]: 
a    0
a    1

In [9]: obj['c']
Out[9]: 4

In [10]: df = DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])

In [11]: df
Out[11]: 
          0         1         2
a  0.512327 -0.198731 -1.271832
a  0.753982  0.513032  0.928941
b -1.565693 -0.052944  0.286924
b -0.861644 -0.114541  0.152211

In [12]: df.ix['b']
Out[12]: 
          0         1         2
b -1.565693 -0.052944  0.286924
b -0.861644 -0.114541  0.152211


In [13]: obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])

In [14]: obj
Out[14]: 
a    0
a    1
b    2
b    3
c    4

In [15]: obj.index.is_unique
Out[15]: False

In [16]: obj['a']
Out[16]: 
a    0
a    1

In [17]: obj['b']
Out[17]: 
b    2
b    3

In [18]: obj['c']
Out[18]: 4

In [19]: df = Dat
DataFrame      DataSource     DateFormatter  DateLocator    

In [19]: df = DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])

In [20]: df
Out[20]: 
          0         1         2
a  0.089692  0.161029  0.806586
a  0.768763  0.892751 -0.498503
b -1.161794 -0.780614 -1.553206
b  2.026748  0.576726 -0.816021

In [21]: df.ix['b']
Out[21]: 
          0         1         2
b -1.161794 -0.780614 -1.553206
b  2.026748  0.576726 -0.816021

#Summarizing and Computing Descriptive statistics 

In [23]: df = DataFrame([[1.4, np.nan], [7.1, -4.5],
   ....: [np.nan, np.nan], [0.75, -1.3]],
   ....: index=['a', 'b', 'c', 'd'],
   ....: columns=['one', 'two'])

In [24]: df
Out[24]: 
    one  two
a  1.40  NaN
b  7.10 -4.5
c   NaN  NaN
d  0.75 -1.3

In [25]: df.dum()
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/Users/Makindo/<ipython-input-25-f4a819757a15> in <module>()
----> 1 df.dum()

/Library/Python/2.7/site-packages/pandas-0.10.1-py2.7-macosx-10.8-intel.egg/pandas/core/frame.pyc in __getattr__(self, name)
   2044             return self[name]
   2045         raise AttributeError("'%s' object has no attribute '%s'" %
-> 2046                              (type(self).__name__, name))
   2047 
   2048     def __setattr__(self, name, value):

AttributeError: 'DataFrame' object has no attribute 'dum'

In [26]: df.sum()
Out[26]: 
one    9.25
two   -5.80

In [27]: df.sum(axis=1)
Out[27]: 
a    1.40
b    2.60
c     NaN
d   -0.55

In [28]: df.means(axis=1, skipna=False)
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/Users/Makindo/<ipython-input-28-352cc97b574c> in <module>()
----> 1 df.means(axis=1, skipna=False)

/Library/Python/2.7/site-packages/pandas-0.10.1-py2.7-macosx-10.8-intel.egg/pandas/core/frame.pyc in __getattr__(self, name)
   2044             return self[name]
   2045         raise AttributeError("'%s' object has no attribute '%s'" %
-> 2046                              (type(self).__name__, name))
   2047 
   2048     def __setattr__(self, name, value):

AttributeError: 'DataFrame' object has no attribute 'means'

In [29]: df.mean(axis=1, skipna=False)
Out[29]: 
a      NaN
b    1.300
c      NaN
d   -0.275

In [30]: df.idxmax()
Out[30]: 
one    b
two    d

In [31]: df.cumsum()
Out[31]: 
    one  two
a  1.40  NaN
b  8.50 -4.5
c   NaN  NaN
d  9.25 -5.8

In [32]: df.describe()
Out[32]: 
            one       two
count  3.000000  2.000000
mean   3.083333 -2.900000
std    3.493685  2.262742
min    0.750000 -4.500000
25%    1.075000 -3.700000
50%    1.400000 -2.900000
75%    4.250000 -2.100000
max    7.100000 -1.300000

In [33]: obj = Series(['a', 'a', 'b', 'c'] * 4)

In [34]: obj.describe()
Out[34]: 
count     16
unique     3
top        a
freq       8

In [35]: 














