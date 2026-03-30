import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object import RigidObjectCfg

from g1_hoi_learning.objects import *

_COMMON_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)

_COMMON_MASS_PROPS = sim_utils.MassPropertiesCfg(density=200.0)


def _object_cfg(usd_path: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        spawn=sim_utils.UsdFileCfg(
            scale=(1.0, 1.0, 1.0),
            usd_path=usd_path,
            activate_contact_sensors=True,
            mass_props=_COMMON_MASS_PROPS,
            rigid_props=_COMMON_RIGID_PROPS,
        ),
    )


CLOTHESSTAND_CFG = _object_cfg(CLOTHESSTAND_USD_PATH)
FLOORLAMP_CFG = _object_cfg(FLOORLAMP_USD_PATH)
LARGEBOX_CFG = _object_cfg(LARGEBOX_USD_PATH)
LARGETABLE_CFG = _object_cfg(LARGETABLE_USD_PATH)
MONITOR_CFG = _object_cfg(MONITOR_USD_PATH)
PLASTICBOX_CFG = _object_cfg(PLASTICBOX_USD_PATH)
SMALLBOX_CFG = _object_cfg(SMALLBOX_USD_PATH)
SMALLTABLE_CFG = _object_cfg(SMALLTABLE_USD_PATH)
SUITCASE_CFG = _object_cfg(SUITCASE_USD_PATH)
TRASHCAN_CFG = _object_cfg(TRASHCAN_USD_PATH)
TRIPOD_CFG = _object_cfg(TRIPOD_USD_PATH)
WHITECHAIR_CFG = _object_cfg(WHITECHAIR_USD_PATH)
WOODCHAIR_CFG = _object_cfg(WOODCHAIR_USD_PATH)
