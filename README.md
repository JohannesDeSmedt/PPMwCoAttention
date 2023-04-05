# PPMwCoAttention

## Setup
The experimental setup was performed thusly:
![Experimental setup](https://github.com/JohannesDeSmedt/PPMwCoAttention/blob/main/img/NSP_RT_overview.png)

## Implementation
All code tested with the following Python (3.9) libraries:
- scikit-learn 1.0.2
- tensorflow 2.4.3
- numpy 1.19.5
- pm4py 2.2.19.1
- seaborn 0.11.2

File [run_experiments_nap.py](run_experiments_nap.py) fully recreates the experimental evaluation.

File [nap_co_attention_mfb_visualization.py](nap_co_attention_mfb_visualization.py) allows to create the attention score graphs for log and trace.

## Datasets
We used 3 datasets, as-is, from the following locations:
- [BPI12](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f)
- [Sepsis](https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460)
- [Italian helpdesk](https://doi.org/10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb)

## Outcomes
Attention scores for individual prefixes can be generated similar to the ones for BPI12 and the Helpdesk log below:

![Attention scores single prefix in BPI 12](https://github.com/JohannesDeSmedt/PPMwCoAttention/blob/main/img/Co-attention-bpi12.png)

![Attention scores single prefix in helpdesk log](https://github.com/JohannesDeSmedt/PPMwCoAttention/blob/main/img/Co-attention-helpdesk.png)
