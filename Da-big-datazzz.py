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









