# Decision-Tree-Regression-implementation-from-scratch

I have implemented a decision tree regression algorithm on a univariate dataset, which contains 272 data points about the duration of the eruption and waiting time between eruptions for the Old Faithful geyser in Yellowstone National Park, Wyoming, USA (https://www.yellowstonepark.com/thingsto-do/about-old-faithful), in the file named dataset.csv.

The algorithm was implemented using the pre-pruning rule, such that if a node has 𝑃 or fewer data points, it is converted into a terminal node. Using pre-pruning rule is one of the methods to prevent over-fitting. 

The following indexing rule was used (since Python starts from index 0) and all data structures were made consistent with this: 
For left node: 2*parent + 1   (instead of 2*parent)
For right node: 2*parent + 2 (instead of 2* parent + 1)

During the learning process, at each step we generate all possible split positions and then pick the best one based on the score function given below:
![score_fx](/score_fx.png)

Based on the selected split, then we generate the left node and the right node. The algorithm goes on until there are no nodes to split or we reach the P value for remaining nodes. 

The decision tree is visualized for P=25 as shown below:
![Pruning=25](/Figure_1n.png)

y_predicted is calculated for X_test and the RMSE is calculated to compare y_predicted and y_test. The error was ~6.45 when P=25. 

The algorithm was then run on varying P values and the corresponding RMSE were compared. We can see from the following graph that RMSE is minimal when P is between 30 and 40. 
![Pruning vs RMSE](/Figure_2n.png)


