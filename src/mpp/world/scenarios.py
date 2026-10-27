"""Practice scenarios: a mission (start -> goal) plus optional boxes falling from the shelves.

A falling box is unknown to the global planner. It hangs above the aisle (out of the lidar
plane) and drops when the robot comes within `trigger_distance` of it, so the robot has
already committed to its plan and has to react: either drive around the box (the corridor
is still wide enough) or plan a new route (the corridor is blocked).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FallingBox:
    name: str
    x: float
    y: float
    size: tuple[float, float, float] = (0.3, 0.3, 0.3)  # full extents, metres
    trigger_distance: float = 1.6  # drop when the robot centre is this close (metres)
    drop_height: float = 1.15  # bottom of the box above the floor before the drop
    mass: float = 1.5


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    start: str
    goal: str
    boxes: tuple[FallingBox, ...] = field(default_factory=tuple)
    description: str = ""


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in [
        Scenario(
            name="open",
            title="Пустой склад",
            start="S3",
            goal="G1",
            description=(
                "Без помех. Прямая к G1 упирается в U-образный тупик, а стеллаж L3 закрывает "
                "выход из стартовой зоны: хорошо видно, как разные планировщики ищут обход."
            ),
        ),
        Scenario(
            name="bypass",
            title="Коробка в широком проходе: объезд",
            start="S1",
            goal="G3",
            boxes=(FallingBox("box_bypass", 5.0, 1.12, size=(0.4, 0.4, 0.3)),),
            description=(
                "Коробка падает в нижний проход шириной 0.9 м и прижимается к стеллажу. "
                "Рядом остаётся около 0.48 м, роботу (0.21 м) хватает: локальный планировщик "
                "должен объехать её, не меняя глобальный маршрут."
            ),
        ),
        Scenario(
            name="blocked",
            title="Коробка в узком коридоре: перестроение маршрута",
            start="S2",
            goal="G2",
            boxes=(FallingBox("box_blocked", 5.0, 3.425, size=(0.36, 0.36, 0.3)),),
            description=(
                "Кратчайший путь идёт через узкий коридор 0.45 м. Коробка перекрывает его "
                "целиком, объехать нельзя: нужно глобальное перепланирование через нижний "
                "или верхний проход."
            ),
        ),
    ]
}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise KeyError(f"unknown scenario {name!r}; known: {', '.join(SCENARIOS)}") from None
