from agents.sac import SACAgent
from agents.dynamics_fb import ForwardBackwardAgent as DynamicsForwardBackwardAgent
from agents.fb import ForwardBackwardAgent
from agents.dynamics_aware_iql import GCIQLAgent

agents = dict(
    sac=SACAgent,
    dynamics_fb=DynamicsForwardBackwardAgent,
    fb=ForwardBackwardAgent,
    gciql=GCIQLAgent
)
