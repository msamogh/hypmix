import pandas as pd
from pgmpy.models import MarkovModel
from pgmpy.estimators import BicScore, HillClimbSearch
from pgmpy.inference import BeliefPropagation
from pgmpy.independencies import Independencies

# Sample dataset
data = {
    'A': ['a1', 'a2', 'a1', 'a2', 'a1'],
    'B': ['b1', 'b1', 'b2', 'b2', 'b1'],
    'C': ['c1', 'c1', 'c2', 'c2', 'c1'],
    'D': ['d1', 'd1', 'd1', 'd2', 'd2']
}
df = pd.DataFrame(data)

# Define the MRF structure
model = MarkovModel()
model.add_nodes_from(['A', 'B', 'C', 'D'])
model.add_edges_from([('A', 'C'), ('C', 'B'), ('A', 'D'), ('D', 'B')])

# Define the conditional independence assertions
independencies = Independencies(['A', 'B', 'C'], ['A', 'D', 'B'], ['C', 'D', 'A', 'B'])
model.add_cpds(independencies)

# Check if the dataset follows the MRF structure
hc = HillClimbSearch(df)
scoring_method = BicScore(df)
best_model = hc.estimate(scoring_method)

# Compare the estimated model with the defined MRF structure
print("Edges in the defined model:", model.edges())
print("Edges in the estimated model:", best_model.edges())

# Perform inference and check conditional independencies
inference = BeliefPropagation(best_model)

# Example conditional independence test
query = inference.query(variables=['A'], evidence={'C': 'c1'})
print(query)
