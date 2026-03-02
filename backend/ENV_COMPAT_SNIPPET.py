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
        "deadhead_cost_per_mile": float(pick("DEADHEAD_COST_PER_MILE", "DEADHEAD_COST", default="2.0")),
        "reassignment_admin_cost": float(pick("REASSIGNMENT_ADMIN_COST", default="10")),
        "emergency_rest_penalty": float(pick("EMERGENCY_REST_PENALTY", default="50")),
        "outsourcing_base_cost": float(pick("OUTSOURCING_BASE_COST", default="200")),
        "outsourcing_per_mile": float(pick("OUTSOURCING_PER_MILE", default="2.0")),
        "overtime_cost_per_minute": float(pick("OVERTIME_COST_PER_MINUTE", default="3.0")),
        "rank_deadhead_miles_weight": float(pick("RANK_DEADHEAD_MILES_WEIGHT", default="1.0")),
        "rank_deadhead_minutes_weight": float(pick("RANK_DEADHEAD_MINUTES_WEIGHT", default="0.15")),
        "rank_overtime_minutes_weight": float(pick("RANK_OVERTIME_MINUTES_WEIGHT", default="2.0")),
        "rank_penalty_append": float(pick("RANK_PENALTY_APPEND", default="30.0")),
        "max_duty_minutes": float(pick("MAX_DUTY_MINUTES", default="780")),
    }
