from typing import TypedDict, List, Tuple

class Vehicle(TypedDict):
    id: str
    start_index: int
    end_index: int
    time_window: Tuple[int,int]
    breaks: List[Tuple[int,int]]

class Order(TypedDict):
    id: str
    index: int
    service_time: int
    time_window: Tuple[int,int]
    priority: int
    pickup_index: int | None
    dropoff_index: int | None