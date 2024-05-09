from cbx.dynamics import (
    ParticleDynamic,
    CBXDynamic,
    CBO,
    CBOMemory,
    PSO,
    CBS,
    PolarCBO,
    QCBO,
)


def dispatch_dynamics(dyn_name: str) -> object:
    """Dispatch the dynamics based on the name.

    Args:
        dyn_name (str): The name of the dynamics.

    Raises:
        ValueError: If the dynamic name is unknown.

    Returns:
        object: The dynamics.
    """
    if dyn_name == "particle_dynamic":
        return ParticleDynamic
    elif dyn_name == "cbx":
        return CBXDynamic
    elif dyn_name == "cbo":
        return CBO
    elif dyn_name == "cbo_memory":
        return CBOMemory
    elif dyn_name == "pso":
        return PSO
    elif dyn_name == "cbs":
        return CBS
    elif dyn_name == "polar_cbo":
        return PolarCBO
    elif dyn_name == "qcbo":
        return QCBO
    else:
        raise ValueError("Unknown function name")


def get_available_dynamics() -> list[str]:
    """Get the available dynamics.

    Returns:
        list[str]: A list of available dynamics.
    """
    return [
        "particle_dynamic",
        "cbx",
        "cbo",
        "cbo_memory",
        "pso",
        "cbs",
        "polar_cbo",
        "q_cbo",
    ]
