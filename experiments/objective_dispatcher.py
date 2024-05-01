from cbx.objectives import (
    three_hump_camel,
    McCormick,
    Rosenbrock,
    Himmelblau,
    Rastrigin,
    Rastrigin_multimodal,
    Ackley,
    Ackley_multimodal,
    accelerated_sinus,
    nd_sinus,
    p_4th_order,
    Quadratic,
    Banana,
    Bimodal,
    Unimodal,
    Bukin6,
    cross_in_tray,
    Easom,
    drop_wave,
    Holder_table,
    snowflake,
    eggholder,
    Michalewicz,
)


def dispatch_objective(func_name):
    print(func_name)
    if func_name == "three_hump_camel":
        return three_hump_camel()
    elif func_name == "mccormick":
        return McCormick()
    elif func_name == "rosenbrock":
        return Rosenbrock()
    elif func_name == "himmelblau":
        return Himmelblau()
    elif func_name == "rastrigin":
        return Rastrigin()
    elif func_name == "rastrigin_multimodal":
        return Rastrigin_multimodal()
    elif func_name == "ackley":
        return Ackley()
    elif func_name == "ackley_multimodal":
        return Ackley_multimodal()
    elif func_name == "accelerated_sinus":
        return accelerated_sinus()
    elif func_name == "nd_sinus":
        return nd_sinus()
    elif func_name == "p_4th_order":
        return p_4th_order()
    elif func_name == "quadratic":
        return Quadratic()
    elif func_name == "banana":
        return Banana()
    elif func_name == "bimodal":
        return Bimodal()
    elif func_name == "unimodal":
        return Unimodal()
    elif func_name == "bukin6":
        return Bukin6()
    elif func_name == "cross_in_tray":
        return cross_in_tray()
    elif func_name == "easom":
        return Easom()
    elif func_name == "drop_wave":
        return drop_wave()
    elif func_name == "holder_table":
        return Holder_table()
    elif func_name == "snowflake":
        return snowflake()
    elif func_name == "eggholder":
        return eggholder()
    elif func_name == "michalewicz":
        return Michalewicz()
    else:
        raise ValueError("Unknown function name")


def get_available_objectives():
    return [
        "three_hump_camel",
        "mccormick",
        "rosenbrock",
        "himmelblau",
        "rastrigin",
        "rastrigin_multimodal",
        "ackley",
        "ackley_multimodal",
        "accelerated_sinus",
        "nd_sinus",
        "p_4th_order",
        "quadratic",
        "banana",
        "bimodal",
        "unimodal",
        "bukin6",
        "cross_in_tray",
        "easom",
        "drop_wave",
        "holder_table",
        "snowflake",
        "eggholder",
        "michalewicz",
    ]
