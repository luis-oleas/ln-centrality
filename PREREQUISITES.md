## Libraries

To get a proper execution of the simulator, it is required to set python version 3.5+. Additionally, it is recommended to create a virtual environment to install the required libraries to execute the program as follows:

```sh
$ python -m pip install virtualenv  
$ virtualenv venv 
$ source venv/bin/activate 
$ python -m pip install --upgrade pip 
$ python -m pip install grpcio 
```

Therefore, the simulator requires the following libraries with its corresponding versions as well as additional code modules:

Name  | Version | Description
------------- | ------------- | -------------
jsonpickle | `1.4.2`	| library used to serialize and deserialize complex Python object to and from JSON, which in this case allows to create output files with the test results
numpy | `1.18.5` | handles the assignation of random balance with any of the following distributions: beta, exponential, normal, uniform
networkx |  `2.4` | library to create and manipulate the structure of dynamic and complex networks
matplotlib | `3.3.4` | library to create plots with data from metrics
pandas | `1.2.3` | library for data analysis and statistics for the metrics of LN
sklearn | `1.3.1` | library to compute RMSE
scipy | `1.6.3` | library to generate Spearman rank correlation

## Additional modules

Package | Subpackage | Module | Description
:--------: | :--------: | :--------: | --------
ln | --- | --- | root in the structure of the program files
.| --> | ln_centrality | module that invokes the functionality of the simulation, i.e. it is the main module in the program
. | --> | utils | module that provides with generic methods, functions and classes used along the whole program
. | --> | data_analysis | module used to set the data to generate the results as figures
. | --> | plots	| module with the required structure to create the plots of the results

 
## Support files

1. parameters.json
   
Key | Sub-key | Description
:-----: | :-----: | ----- 
loop | --- | number of repetitions executed of query route implementation over the same couple of nodes
num_k | --- | number of routes to gather by means of the Yen's algorithm
sleep | --- | seconds that the simulator halts previous to continue with a payment
update | --- | parameter considered for a future implementation
weight | --- | parameter that defines the edge attribute to measure the metrics betweenness flow, betweenness centrality and page rank. The possible options are between: `capacity` and `fee`
k_fraction | --- | number to calculate the fraction of nodes for k of betweenness centrality
num_routes | --- | number of routes to simulate query routes that will be considered at the time to create a test.json file
max_amount | --- | max payment amount to send to a destiny node
step_diff_ns | --- | increment in nanoseconds at the moment to calculate a timeout. Default 0.5 seconds
min_diff_ns | --- | nanoseconds specifying at which position to start. Default 1 second
max_diff_ns | --- | nanoseconds specifying at which position to start. Default 2.5 seconds
test_file | --- | name of the json file that contains the test set
results_file | --- | name of the json file that contains the results of the tests. `Deprecated`
metrics_file | --- | name of the json file that contains the metrics of the nodes.
polar_path | --- | Parameter that helps to configure the connection to a node LND
alpha_strength | --- | tuning parameter to calculate the degree strength measure of nodes
alpha_centrality | --- | tuning parameter to calculate the degree centrality measure of nodes
balance | --- |  There is enough balance in all the channels to fulfill the payment.
length_path | --- | The length of the path is smaller or equal than 20. `Deprecated`
is_length_path | --- |Flag that controls whether or not find longest paths `Deprecated`
amount_payment | --- | The amount of the payment is higher than the minimum (σ > min_htlc).
num_pending_htlc | --- | The number of existing HTLCs in each channel is less than 14.
is_rand_htlc | --- | Flag that indicates whether or not the assignation of htlc is random
flag_length_path | --- | Flag that enables/disables restriction 6.1.1 The length of the path is smaller or equal than 20
flag_channel_enabled | --- | Flag that enables/disables restriction 6.1.0 Channel enabled
flag_enough_balance | --- | Flag that enables/disables restriction 6.1.2 There is enough balance in all the channels to fulfill the payment
flag_minimum_htlc | --- | Flag that enables/disables restriction 6.1.3 The number of existing HTLCs in each channel is less than 14
flag_valid_timeout | --- | Flag that enables/disables restriction 6.1.4.1 There exists a set of timeouts for all HTLCs in the path that fulfill the conditions on the nodes policies for all nodes in the path
flag_payment_greater_delta | --- | Flag that enables/disables restriction 6.1.4.2 The amount of the payment is higher than the minimum (σ>min_htlc)
flag_flow_betweenness | --- | Flag that enables/disables the calculation of flow betweenness
flag_current_flow_betweenness | --- | Flag that enables/disables the calculation of current flow betweenness
pivot_paths | --- | The searching boundary of paths to be used for the flow betweenness metric
penalty_fee | --- | Penalty given to those policies that contain null values regarding to fee_base_msat `Deprecated`
is_payment | --- | flag that allows the run or not of payments
flag_metrics | --- | flag that activates the calculation of metrics
