DeepWalk: Online Learning of Social Representations
===================================================

This is a python implementation of *DeepWalk* by Bryan Perozzi. This is a contribution for Dr. Guozhu Dong's repository which is a collection of Feature Engineering projects for his new textbook called **'Feature Engineering'** [currently unplublished]. 

**Dataset** : *BlogCatalog* 

Download here:
[mat file](http://leitang.net/code/social-dimension/data/blogcatalog.mat)

- Number of users : 10,312
- Number of friendships/edges : 333,983
- Number of groups to which users can subscribe to : 39

Install pre-reqs by running: 
`pip install -r req.txt`


Run the code by entering: 
`python DeepWalk.py`


A Full list of cmd line arguments are shown by entering: 
```
python DeepWalk.py -h
```

```
usage: DeepWalk [-h] [--d D] [--walks WALKS] [--len LEN] [--window WINDOW]
                [--hs HS] [--lw LW] [--le LE] [-w] [-e] [-o]

Implementation of DeepWalk model. File Author: Apoorva

optional arguments:
  -h, --help       show this help message and exit
  --d D            Dimensions of word embeddings
  --walks WALKS    Number of walks per node
  --len LEN        Length of random walk
  --window WINDOW  Window size for skipgram
  --hs HS          0 - Negative Sampling 1 - Hierarchical Softmax
  --lw LW          Load random walk corpus from file
  --le LE          Load embeddings from file
  -w               Flag to save random walk corpus to disk
  -e               Flag to save word embeddings to disk
  -o               Flag to save Console output to disk
```


Some sample walk corpus files and embeddings are included in the repository. Feel free to download and play with them.


Here is an example of how you would pass arguments from the terminal:
```
python DeepWalk.py --d 128 --walks 40 --len 80 --window 10 
```
Default values are used if no arguments are passed. Here are the default parameter values:

>Dimensions = 128
>
>Walks per node = 10
>
>Walk length = 30
>
>Window size for skip-gram = 5
>
>Heirarchical Softmax = True

*A Jupyter notebook version of the implementation titled DeepWalk.ipynb has also been included in the repository.*


The report **DeepWalkReport.pdf** contains the tabulated results along with comparisons to other models and explanation of the concepts behind DeepWalk.
If you want to take a look at the raw output dump of the all the test runs, they can be found at *OutputDump.txt*  

 
**References:**

[DeepWalk: Online Learning of Social Representations](http://dl.acm.org/citation.cfm?id=2623732)

[Video: Presentation of DeepWalk by B. Perozzi at KDD 2014](https://www.youtube.com/watch?v=n12HS-24CtA)

[Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases)


*Feel free to contact me at apoorva.v94@gmail.com or [HERE](https://linkedin.com/in/apoorvavinod)*