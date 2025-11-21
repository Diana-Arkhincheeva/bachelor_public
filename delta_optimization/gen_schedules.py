from dataclasses import dataclass
import json
import logging
import statistics
from typing import List

from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions.parallel_machine_utilities import (
    calculate_twct,
)
from privacyschedulingtools.total_weighted_completion_time.pup.solver.pup_parallel import (
    PupParallel,
)
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import (
    ParallelSchedulingParameters,
)
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import (
    ParallelScheduleFactory,
    ParallelSchedule,
)
from privacyschedulingtools.total_weighted_completion_time.entity.domain import (
    IntegerDomain,
)
from privacyschedulingtools.total_weighted_completion_time.entity.adversary.parallel_adversary import (
    ParallelLocalWSPTAdversary,
)

from concurrent.futures import ProcessPoolExecutor


"""Main Script für Kalendergenerierung:
Generiert schedules mit dem mitterwert epsilon-werte>=0.5"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def schedule_to_json(schedule: ParallelSchedule, epsilon: float, twct: float) -> dict:
    machines = []

    for machine_id, jobs in enumerate(schedule.allocation):
        job_list = []
        for scheduled_job in jobs:
            job = schedule.jobs[scheduled_job.id]
            job_list.append(
                {
                    "job_id": job.id,
                    "start_time": scheduled_job.start_time,
                    "processing_time": job.processing_time,
                    "weight": job.weight,
                }
            )
        machines.append({"machine_id": machine_id, "jobs": job_list})

    return {"epsilon": round(epsilon, 4), "twct": round(twct, 2), "machines": machines}


def median_epsilon_for_schedule(
    schedule: ParallelSchedule,
    params: ParallelSchedulingParameters,
    epsilon_threshold: float,
    n_attacks: int = 10,
) -> float:
    adversary = ParallelLocalWSPTAdversary(params)
    epsilons: List[float] = []

    for i in range(n_attacks):
        attack_result = adversary.execute_attack(schedule)
        solver = PupParallel(
            schedule=schedule,
            privacy_threshold=epsilon_threshold,
            utility_threshold=1.0,
            schedule_parameters=params,
        )
        privacy_loss, _ = solver.calculate_privacy_loss(schedule, attack_result)
        epsilon = max(privacy_loss) if privacy_loss else float("inf")
        epsilons.append(epsilon)
        logger.info(f"Attack {i + 1}/{n_attacks}: ε = {epsilon:.4f}")

    median_eps = statistics.median(epsilons)
    logger.info(f"Median ε nach {n_attacks} Attacken: {median_eps:.4f}")
    return median_eps


@dataclass
class Evaluator:
    adversary: ParallelLocalWSPTAdversary
    epsilon_threshold: float
    params: ParallelSchedulingParameters

    def eval_schedule(
        self, schedule: ParallelSchedule
    ) -> tuple[ParallelSchedule, float, int]:
        logger.info("Start")
        attack_result = self.adversary.execute_attack(schedule)

        solver = PupParallel(
            schedule=schedule,
            privacy_threshold=self.epsilon_threshold,
            utility_threshold=1.0,
            schedule_parameters=self.params,
        )

        privacy_loss, _ = solver.calculate_privacy_loss(schedule, attack_result)
        epsilon: float = max(privacy_loss) if privacy_loss else float("inf")
        twct = calculate_twct(schedule)

        logger.info(f"Result: ε = {epsilon:.4f}, TWCT = {twct:.2f}")
        return (schedule, epsilon, twct)


def generate_balanced_schedule(
    epsilon_threshold: float,
    params: ParallelSchedulingParameters,
    max_attempts: int = 10,
    json_filename: str = "bad_epsilon_50_3.json",
):
    factory = ParallelScheduleFactory(params)
    adversary = ParallelLocalWSPTAdversary(params)

    valid_schedules = []

    logger.info(f"Looking for one schedule or more with ε-value >= {epsilon_threshold}")

    schedules = (
        factory.generate_random_schedule_with_dispatching_rule()
        for _ in range(max_attempts)
    )
    evaluator = Evaluator(adversary, epsilon_threshold, params)

    with ProcessPoolExecutor() as executor:
        for schedule, epsilon, twct in executor.map(evaluator.eval_schedule, schedules):
            if epsilon >= epsilon_threshold:
                # считаем медиану по 10 атакам
                median_eps = median_epsilon_for_schedule(
                    schedule, params, epsilon_threshold, n_attacks=10
                )
                logger.info(
                    f"THERE IS ONE POSSIBLE SCHEDULE WITH median ε >= {median_eps:.4f}"
                )
                valid_schedules.append(schedule_to_json(schedule, median_eps, twct))

    if valid_schedules:
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(valid_schedules, f, indent=4, ensure_ascii=False)
        logger.info(
            f"{len(valid_schedules)} schedules wurden in {json_filename} gespeichert."
        )
    else:
        logger.warning("There is no matching schedule.")


def main():
    params = ParallelSchedulingParameters(
        job_count=50,
        machine_count=3,
        processing_time_domain=IntegerDomain(10, 60),
        weight_domain=IntegerDomain(1, 5),
    )

    epsilon_threshold = 0.5
    generate_balanced_schedule(epsilon_threshold, params, max_attempts=20)


if __name__ == "__main__":
    main()
