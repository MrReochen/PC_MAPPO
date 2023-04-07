from .mappo import R_MAPPOPolicy
from .con_mappo import CON_MAPPOPolicy

Policies = {}

Policies['mappo'] = R_MAPPOPolicy
Policies['con_mappo'] = CON_MAPPOPolicy