from agents.sac import SACAgent
from agents.fb import ForwardBackwardAgent
from agents.dynamics_aware_iql import GCIQLAgent
agents = dict(
    sac=SACAgent,
    fb=ForwardBackwardAgent,
    gciql=GCIQLAgent
)
