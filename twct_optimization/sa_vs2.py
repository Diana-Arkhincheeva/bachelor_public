from dataclasses import dataclass
import logging
import random
import math
import json
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
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
from privacyschedulingtools.total_weighted_completion_time.entity.job import Job
from privacyschedulingtools.total_weighted_completion_time.entity.adversary.parallel_adversary import (
    ParallelLocalWSPTAdversary,
)
from privacyschedulingtools.total_weighted_completion_time.entity.domain import (
    IntegerDomain,
)
from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions.parallel_machine_utilities import (
    calculate_twct,
)
from privacyschedulingtools.total_weighted_completion_time.pup.util.transformations import (
    Transformation,
)


""" SA mit bis zu 20 Nachbarn"""
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(logging.FileHandler("sa_vs2_30_3.log"))


def load_schedules_from_json(
    file_path: str,
) -> List[Tuple[ParallelSchedule, ParallelSchedulingParameters]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    schedules: List[Tuple[ParallelSchedule, ParallelSchedulingParameters]] = []
    for schedule_data in data:
        job_map = {}
        machines_data = schedule_data["machines"]
        machine_count = len(machines_data)

        for machine in machines_data:
            for job_info in machine["jobs"]:
                job_id = job_info["job_id"]
                if job_id not in job_map:
                    job_map[job_id] = Job(
                        job_id, job_info["processing_time"], job_info["weight"]
                    )

        jobs = list(job_map.values())
        processing_times = [job.processing_time for job in jobs]
        weights = [job.weight for job in jobs]

        params = ParallelSchedulingParameters(
            job_count=len(jobs),
            machine_count=machine_count,
            processing_time_domain=IntegerDomain(
                min(processing_times), max(processing_times)
            ),
            weight_domain=IntegerDomain(min(weights), max(weights)),
        )

        factory = ParallelScheduleFactory(params)
        schedule = factory.generate_schedule_from_jobs(
            jobs, machine_count, parameters=params, schedule_type="wspt"
        )

        schedules.append((schedule, params))
    return schedules


@dataclass
class Annealer:
    epsilon_threshold: float
    params: ParallelSchedulingParameters

    def calculate_metrics(self, schedule: ParallelSchedule) -> Tuple[float, float]:
        adversary = ParallelLocalWSPTAdversary(self.params)
        attack_result = adversary.execute_attack(schedule)
        solver = PupParallel(
            schedule,
            privacy_threshold=self.epsilon_threshold,
            utility_threshold=1.0,
            schedule_parameters=self.params,
        )
        privacy_loss, _ = solver.calculate_privacy_loss(schedule, attack_result)
        epsilon = max(privacy_loss) if privacy_loss else float("inf")
        twct = calculate_twct(schedule)
        return epsilon, twct

    def get_neighbors(
        self,
        schedule: ParallelSchedule,
        transformations: List[Transformation],
        limit: int = 20,
    ) -> List[ParallelSchedule]:
        solver = PupParallel(
            schedule,
            privacy_threshold=self.epsilon_threshold,
            utility_threshold=1.0,
            schedule_parameters=self.params,
        )
        neighbors = []
        for t in transformations:
            try:
                if t == Transformation.MOVE:
                    neighbors.extend(solver._add_move_neighbors(schedule))
                elif t == Transformation.SWAPPING_JOBS:
                    neighbors.extend(solver._add_swap_neighbors(schedule))
                elif t == Transformation.SWAP_PROC:
                    neighbors.extend(solver._add_swap_neighbors(schedule))
                    neighbors.extend(solver._add_processing_time_neighbors(schedule))
                elif t == Transformation.SWAP_ALL:
                    neighbors.extend(solver._add_swap_all(schedule))
                elif t == Transformation.MOVE_PROC:
                    neighbors.extend(solver._add_move_neighbors(schedule))
                    neighbors.extend(solver._add_processing_time_neighbors(schedule))
                elif t == Transformation.ALT_MOVE_PROC:
                    neighbors.extend(solver._add_move_neighbors_alt(schedule))
                    neighbors.extend(solver._add_processing_time_neighbors(schedule))
                elif t == Transformation.ALT_MOVE:
                    neighbors.extend(solver._add_move_neighbors_alt(schedule))
                elif t == Transformation.SWAP_ALL_PROC:
                    neighbors.extend(solver._add_swap_all(schedule))
                    neighbors.extend(solver._add_processing_time_neighbors(schedule))
            except Exception as e:
                logger.error(f"Neighbor error {t.name}: {str(e)}")

        random.shuffle(neighbors)
        if len(neighbors) > limit:
            neighbors = neighbors[:limit]

        return neighbors

    def simulated_annealing(
        self,
        initial_schedule: ParallelSchedule,
        transformations: List[Transformation],
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        min_temp: float = 0.1,
        max_iterations: int = 50,
    ) -> Tuple[float, float, ParallelSchedule]:
        current = initial_schedule
        best = current
        epsilon, twct = self.calculate_metrics(current)
        best_epsilon, best_twct = epsilon, twct

        logger.info(f"SA_vs2 start: ε={epsilon:.4f}, TWCT={twct:.2f}")

        temperature = initial_temp

        for i in range(1, max_iterations + 1):
            neighbors = self.get_neighbors(current, transformations, limit=20)
            if not neighbors:
                temperature *= cooling_rate
                if temperature < min_temp:
                    break
                continue

            improved = False
            for candidate in neighbors:
                cand_epsilon, cand_twct = self.calculate_metrics(candidate)
                if cand_epsilon > self.epsilon_threshold:
                    continue

                delta = cand_twct - twct
                accept = delta < 0 or random.random() < math.exp(
                    -delta / max(temperature, 1e-12)
                )
                if accept:
                    current = candidate
                    epsilon, twct = cand_epsilon, cand_twct
                    improved = True

                if cand_epsilon <= self.epsilon_threshold and cand_twct < best_twct:
                    best = candidate
                    best_epsilon, best_twct = cand_epsilon, cand_twct
                    logger.info(
                        f"!!!UPDATE: ε={best_epsilon:.4f}, TWCT={best_twct:.2f}!!!"
                    )

            if not improved:
                temperature *= cooling_rate
            if temperature < min_temp:
                break

        return best_epsilon, best_twct, best


def run_sa_for_schedule(args):
    schedule, params, idx = args
    annealer = Annealer(epsilon_threshold=0.5, params=params)

    transformations = [
        Transformation.MOVE,
        Transformation.SWAPPING_JOBS,
        Transformation.SWAP_PROC,
        Transformation.SWAP_ALL,
        Transformation.MOVE_PROC,
        Transformation.SWAP_ALL_PROC,
        Transformation.ALT_MOVE_PROC,
        Transformation.ALT_MOVE,
    ]

    logger.info(f"SA gestartet für Kalender {idx}")
    best_epsilon, best_twct, best_schedule = annealer.simulated_annealing(
        initial_schedule=schedule,
        transformations=transformations,
        max_iterations=50,
    )
    logger.info(
        f"SA fertig für Kalender {idx}, ε={best_epsilon:.4f}, TWCT={best_twct:.2f}"
    )

    result_dict = {
        "index": idx,
        "best_epsilon": round(best_epsilon, 4),
        "best_twct": round(best_twct, 2),
        "machines": [],
    }
    for machine_id, jobs in enumerate(best_schedule.allocation):
        machine_jobs = []
        for scheduled_job in jobs:
            job = best_schedule.jobs[scheduled_job.id]
            machine_jobs.append(
                {
                    "job_id": job.id,
                    "start_time": scheduled_job.start_time,
                    "processing_time": job.processing_time,
                    "weight": job.weight,
                }
            )
        result_dict["machines"].append({"machine_id": machine_id, "jobs": machine_jobs})

    return result_dict


def main():
    input_file = "new_balanced_schedules_30_3.json"
    schedules = load_schedules_from_json(input_file)
    tasks = [(s, p, i) for i, (s, p) in enumerate(schedules)]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_sa_for_schedule, tasks))

    output_file = "sa_vs2_30_3.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Alle besten Ergebnisse gespeichert in {output_file}")
    for r in results:
        logger.info(
            f"[{r['index']}] ε={r['best_epsilon']:.4f}, TWCT={r['best_twct']:.2f}"
        )


if __name__ == "__main__":
    main()
