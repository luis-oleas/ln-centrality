# ln_centrality
LN_Centrality is a simulator that works over the lightning network with the aim to provide a global understanding about its functionality as a network and possible improvement to its technology.
The simulator obtains data from the network topology as the main input for the calculation of centrality metrics. By default, each node provides its own configurations as well as its connection with other nodes through payment channels.
As a premise, the simulator calculates on unrestricted (G<sub>1</sub>) and restricted (G<sub>2</sub>) data from the network to contrast the results with or without restrictions.

Initially, the data is filtered through populate_graphs, which considers only those channels with both policies enabled. A partir de esos datos no restringidos, el simulador calcula todas las métricas. Sin embargo, para calcular las mismas métricas en datos restringidos, el simulador hace uso de get_graphs_restrictions para filtrar los datos de acuerdo con las restricciones definidas. 

Previous to execute the simulator, the reader must fulfill the following [Prerequisites](PREREQUISITES.md) to be able to execute the software properly.

* The section [Libraries](./PREREQUISITES.md#libraries) specifies the several libraries that must be installed beforehand the execution of the simulator
* The section [Additional modules](./PREREQUISITES.md#additional-modules) makes reference to all the modules involved in the simulator and describe its functionality
* The section [Support files](./PREREQUISITES.md#support-files) explains the essential parameters for the proper execution of the simulator.

To sum up, the reader must perform the following steps previously to execute the simulator:

1. Download the [ln_model](https://github.com/StvanLeo/ln_centrality.git) github repository
2. Create a python virtual environment on the same folder as the one that contains the github repository
3. Install the required libraries with their corresponding versions
4. Set the values of the configuration parameters on `parameters.json`
8. Recreate, if necessary, the snapshot file (lnd_describegraph_regtest.json) through the following command on a terminal:
```sh
  root@alice:/# su lnd
  $ lncli --network=regtest describegraph
```

## Test

The following sequence diagram shows the flow of the simulator

<!---![Test](http://www.plantuml.com/plantuml/png/XLJ1Sfmm3Btp5RfrSsdlxA59RvrfPZ9DBvtPYu85k34ojKKs_7rbGnZSakaUTW2_z_Ia9xYDWa6cmLLLlgegsyBfcqS31WMX3Nw02pyoZh7tyla6f2U6qqpnfWAetORqhUBO6ugVcXwPoSKBPph2h-WPMkled3WT2HXgSN828mOSI2X4g21f87JsXHXKIT4gGd1YdsebOr6fh8ICp9WtG-WiPajQ8A6-FW0Q4oX6nQvgs-7eW-mVBsLe66LE4iJ6jjNoh_Z2vSPATkwH9tJmWwByDHY0Qs-TYdwdtvCU9xTqXGUS1s85sxY3Pb-F9F22Ji6vkS5FhAm8Lt9E9wjNFjZECY3hn5NIyfojn7CfyZHlq_L14Q9izPQnshcppRmqj2CUcWr-4Zgm2h24SJTQK7oOLsGvWV9NDvwP6W7nQt3dVp7qIeqAWMEp5owHsqgyp_z9_242kiu7A_rGyfTO2rv7ibI2g-AXzsCiEUiIvEmWIfWQcOBwscdc2PQ-wY_EDyzll5rAz_XCvUDIe6onreYRO9y8KeeoZtdxXUBAhGEXoVSVY_TiI_MElJ83s65qb6gYMCzBiFHgTNUhVQmZXTQ78Il2gNNOtRn5gvsJWezppVOkN5OvY7frLMnOOQJGNzqSxqEPuF7PSNRvBRem6WCscrVzwOZzT9pX-wcNuHKRTy1ORrVzU3YBUlZFlYnjuWijCuYfdq1nSiL6hD-pfYTjgF4XQ6oyVXzJT1gS3q86ke1ZVwwY-B6gRfKs3UuF)--->
![Test](http://www.plantuml.com/plantuml/png/bOynJWGn34NxdCBbRWFzRi4UW8vs6clpCmaazYZsBEBsJ4QCaEZeOldzv_oRENTVMmtsKMJXpesQYjvprmRolYBiI0WbqUbLEL9aPqhRVXA3aoPw4ruigzlBcJn3wFG5EmUYIpq20vj4DCAUxvb5K6sIOI6cf3MYJ_9PF9CTiclO6D19z1-gTrEXL2gsjCIImcM6KqJYjJhj1U7eDYHCStb0QAYkuMBu2nmBVe0Ql9UxaUlnVFgzDQu8EZksULuxZFOkdd5yUP8n_4Il8PZh4cwGkBRw3G00)

<!---
```puml
@startuml
skinparam monochrome true
start

if (Load data from\nset of Snapshots?) then (yes)
  :load data of \nnodes/channels\nfrom a folder\nwith snapshots;
else (no)
  :load data of \nnodes/channels\nfrom a json file;
endif
:get default parameters of a given node;
:set parameters of node;
:save metrics to metrics.json file;
:generate figures with the data of\nthe metrics;

stop
@enduml
```
--->