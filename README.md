# IMBandits

IMBandit.py -- Simulator.

egreedy.py -- epsilon-greedy and UCB1 exploration strategy.

degreeDiscount.py, generalGreedy.py -- Two different oracles (IM algorithm).

IC/IC.py -- Independent cascade model, runIC() returns influence result given seed nodes.

### Result

Result on small graph: 2000 nodes

<p float="left">
<img src="./SimulationResults/AvgReward_Diffusion.png" alt="alt text" width="400" height="300">
<img src="./SimulationResults/AcuReward_Diffusion.png" alt="alt text" width="400" height="300">
</p>


#### Parameter

```python
graph_address = './datasets/Flickr/Small_Final_SubG.G'
prob_address = './datasets/Flickr/Probability.dic'

dataset = 'Flickr' #Choose from 'default', 'NetHEPT', 'Flickr'
batchSize = 1
alpha_1 = 0.1
alpha_2 = 0.1 
lambda_ = 0.4
gamma = 0.1
dimension = 4
seed_size = 300
iterations = 200

oracle = degreeDiscountIAC3
```

#### Experiment

Result on Large Graph: 10000+ nodes

<p float="left">
<img src="./SimulationResults/avgReward-dense.png" alt="alt text" width="400" height="300">
<img src="./SimulationResults/acuReward-dense.png" alt="alt text" width="400" height="300">
</p>
Result on two-cluster graph:

<p float="left">
<img src="./SimulationResults/avgReward-cluster.png" alt="alt text" width="400" height="300">
<img src="./SimulationResults/acuReward-cluster.png" alt="alt text" width="400" height="300">
</p>

