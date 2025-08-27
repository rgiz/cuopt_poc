import os

def read_cost_env_defaults():
    def pick(*names, default=None):
        for n in names:
            v = os.getenv(n)
            if v not in (None, ""):
                return v
        return default

    return {
        "delay_cost_per_minute": float(pick("DELAY_COST_PER_MIN", "DELAY_COST_PER_MINUTE", default="10")),
        "deadhead_cost_per_mile": float(pick("DEADHEAD_COST_PER_MILE", "DEADHEAD_COST", default="1.0")),
        "reassignment_admin_cost": float(pick("REASSIGNMENT_ADMIN_COST", default="10")),
        "emergency_rest_penalty": float(pick("EMERGENCY_REST_PENALTY", default="50")),
        "outsourcing_base_cost": float(pick("OUTSOURCING_BASE_COST", default="200")),
        "outsourcing_per_mile": float(pick("OUTSOURCING_PER_MILE", default="2.0")),
        "overtime_cost_per_minute": float(pick("OVERTIME_COST_PER_MINUTE", default="1.0")),
    }
